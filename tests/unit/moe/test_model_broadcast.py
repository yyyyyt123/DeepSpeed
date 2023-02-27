import torch
import deepspeed
import pytest
import deepspeed.comm as dist 

from deepspeed.utils import groups
from unit.common import DistributedTest
from unit.simple_model import SimpleDynamicMoEModel, sequence_dataloader
from unit.util import required_torch_version


@pytest.mark.parametrize("ep_size", [4])
@pytest.mark.parametrize("use_residual", [True])
class TestModel(DistributedTest):
    world_size=4
    
    def test(self, ep_size, use_residual):
        if not required_torch_version():
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 8,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            }
        }
        hidden_dim = 16

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleDynamicMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False,
                                              dynamic_expert_placement=True)
        #dist_init_required=False -- parameterize to True/False?

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            


@pytest.mark.parametrize("ep_size", [4])
@pytest.mark.parametrize("use_residual", [True])
@pytest.mark.parametrize("num_experts", [4])
class TestBroadcast(DistributedTest):
    world_size=4
    
    def _compare_linear_params(self, model):
        params = [p for p in model.linear.parameters()]
        # print(f"rank:{dist.get_rank()}, param:{params}")
        for p in params:     
            # print(p)
            linear_param_list=[ torch.empty_like(p) for _ in range(dist.get_world_size())]
            dist.all_gather(linear_param_list, p)
            trace = linear_param_list[0]
            for i in range(len(linear_param_list)):
                assert trace.equal(linear_param_list[i])
    
    def _compare_expert_params(self, model):
        experts1_model = model.linear2.deepspeed_moe.experts
        experts2_model = model.linear3.deepspeed_moe.experts
        group = groups._get_dynamic_expert_parallel_group_dict()
        expert_params={}
        for key in group.keys():
                expert_params[key] = []
                
        for p in experts1_model.parameters():
            expert_params[p.group_name].append(p)
        for p in experts2_model.parameters():
            expert_params[p.group_name].append(p)
            
        # print(f"rank:{dist.get_rank()}, param:{params}")
        
        for name, params_group in expert_params.items():
            print(f"name:{name} params_group:{params_group}")
            
            _group = groups._get_dynamic_expert_parallel_group(name)
            for p in params_group:
                res = torch.clone(p)
                res = res.mul_(1.0/ dist.get_world_size(_group))
                dist.all_reduce(res, torch.distributed.ReduceOp.SUM, group=_group)
                assert res.equal(p), f"res:{res}, p:{p}"
    
    def test(self, ep_size, use_residual, num_experts):
        rank = dist.get_rank()
        if not required_torch_version():
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 8,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            }
        }
        hidden_dim = 16

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleDynamicMoEModel(hidden_dim, num_experts=4, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False,
                                              dynamic_expert_placement=True)
        #dist_init_required=False -- parameterize to True/False?

        ''' compare linear layer params'''
        self._compare_linear_params(model)
        
        ''' compare experts params'''
        self._compare_expert_params(model)
                
            
        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
         
            ''' after all_reduce of grads, compare params'''
            self._compare_linear_params(model)
            self._compare_expert_params(model)
            
        
        assert False