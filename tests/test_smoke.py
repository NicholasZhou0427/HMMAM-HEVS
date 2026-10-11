from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from WandbOption.MyOption import VRPOption
from Trans_new import (
    Trans_Model_Encoder, Trans_Model_Decoder_low, Trans_Model_Decoder_high,
    Trans_Model_low, Trans_Model_high,
)
from baseline_new import RolloutBaseline_new
from data.Hevrpstw_data import HEVRPSTWDatasetFromFile

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / 'data_high/c20v3_20260601_141156_val.pt'


@pytest.fixture(scope='session', autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)


@pytest.fixture(scope='module')
def batch():
    dataset = HEVRPSTWDatasetFromFile([str(VAL)])
    return next(iter(DataLoader(dataset, batch_size=2)))


def small_options():
    opts = VRPOption(
        OptionID='smoke', DataTime='smoke', customer_size=20,
        embedding_dim=32, n_encode_layers=1, n_heads=4, FF_dim=64,
        hc_decoder_FF_dim=64, fusion_hidden_dim=16, batch_size=2,
        pomo_samples=2, pomo_mode='batch', baseline_mode='scst',
        is_HR_low=True, is_HR_high=False,
    )
    opts.graph_size = 41
    opts.VEHICLE_num = 3
    opts.device = torch.device('cpu')
    return opts


def models():
    torch.manual_seed(3407)
    low_opts = small_options()
    low = Trans_Model_low(
        Trans_Model_Encoder(32, 1, 4, 64, low_opts),
        Trans_Model_Decoder_low(32, low_opts), low_opts,
    )
    high_opts = replace(low_opts, is_HR_low=False, is_HR_high=True)
    high_opts.graph_size = 41
    high_opts.VEHICLE_num = 3
    high_opts.device = torch.device('cpu')
    high = Trans_Model_high(
        Trans_Model_Encoder(32, 1, 4, 64, high_opts),
        Trans_Model_Decoder_high(32, high_opts), high_opts, low,
    )
    return low, high, high_opts


def test_existing_validation_data_loads():
    dataset = HEVRPSTWDatasetFromFile([str(VAL)])
    assert len(dataset) == 2000
    assert dataset[0]['loc'].shape == (41, 2)
    assert set(dataset[0]) == {
        'loc', 'demand', 'start_time', 'end_time', 'service_time', 'mid_time'
    }
    assert all(torch.isfinite(value).all() for value in dataset[0].values())


def test_low_model_runs_on_cpu(batch):
    low, _, _ = models()
    low.decoder.set_decode_type('greedy')
    with torch.no_grad():
        cost, node_ll, vehicle_ll, routes, info = low(batch)
    assert cost.shape == (2,)
    assert routes.shape[:2] == (2, 3)
    assert torch.isfinite(cost).all()
    assert torch.isfinite(node_ll).all()
    assert torch.isfinite(vehicle_ll).all()
    assert routes.min() >= 0 and routes.max() < 41


def test_high_model_reward_matches_cost_and_backpropagates(batch):
    _, high, _ = models()
    high.decoder.set_decode_type('sampling')
    cost, node_ll, vehicle_ll, _, info = high(batch)
    assert torch.isfinite(cost).all()
    torch.testing.assert_close(info['step_costs'].sum(1), cost)
    loss = (cost.detach() * (node_ll + vehicle_ll)).mean()
    loss.backward()
    grads = [p.grad for p in high.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    assert all(p.grad is None for p in high.model_low.parameters())


def test_actual_training_batch_updates_parameters(batch):
    from train import train_batch
    _, high, opts = models()
    high.decoder.set_decode_type('sampling')
    baseline = RolloutBaseline_new(high, opts)
    optimizer = torch.optim.Adam([p for p in high.parameters() if p.requires_grad], lr=1e-4)
    before = {n: p.detach().clone() for n, p in high.named_parameters() if p.requires_grad}
    for batch_id in range(2):
        result = train_batch(high, baseline, optimizer, opts, batch, batch_id, opts.device)
        assert torch.isfinite(result[0]).all()
        assert torch.isfinite(result[1]).all()
    assert result[5] is not None and result[5] > 0
    assert any(not torch.equal(before[n], p) for n, p in high.named_parameters() if n in before)
    assert all(p.grad is None for p in high.model_low.parameters())


def test_checkpoint_roundtrip_and_cli_import(batch, tmp_path):
    from main import save_checkpoint, load_model_checkpoint
    _, high, opts = models()
    high.eval()
    high.decoder.set_decode_type('greedy')
    with torch.no_grad():
        expected = high(batch)[0]
    checkpoint = tmp_path / 'high.pt'
    save_checkpoint(checkpoint, high, opts)
    restored = load_model_checkpoint(checkpoint, torch.device('cpu'))
    restored.eval()
    restored.decoder.set_decode_type('greedy')
    with torch.no_grad():
        actual = restored(batch)[0]
    torch.testing.assert_close(actual, expected)


def test_missing_dataset_fails_without_online_fallback(tmp_path):
    from functions import generate_data
    opts = small_options()
    opts.train_data_path = [str(tmp_path / 'absent.pt')]
    with pytest.raises(FileNotFoundError):
        generate_data(opts, is_val=False)
    opts.train_data_path = None
    with pytest.raises(ValueError, match='train_data_path'):
        generate_data(opts, is_val=False)
