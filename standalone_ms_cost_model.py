# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Callable
import re
import numpy as np
import glob
from tqdm import tqdm
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.logging import get_logger


logging.basicConfig(level=logging.INFO)
get_logger("xgb_model").setLevel(logging.INFO)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR.parent

def create_cost_model():
    # TODO: support other extractions and models
    num_warmup_samples = 0
    extractor = ms.feature_extractor.PerStoreFeature()
    cost_model = ms.cost_model.XGBModel(extractor=extractor, num_warmup_samples=num_warmup_samples)
    print("cost_model", cost_model, dir(cost_model))
    return cost_model


def generate_tasks(samples):
    tasks = []
    candidates = []
    results = []
    target = tvm.target.Target("c")
    space_generator = None
    search_strategy = None
    for sample in samples:
        mod, runtime = sample
        print("mod", mod, type(mod))
        print("runtime", runtime, type(runtime))
        ctx = ms.tune_context.TuneContext(
            mod=mod,
            target=target,
            # space_generator=space,
            # search_strategy=strategy,
            # task_name=task_name,
            # logger=logger,
            # rand_state=rand_state,
            # num_threads=num_tuning_cores,
        )
        tasks.append(ctx)
        sched = tvm.tir.Schedule(mod)
        candidate = ms.MeasureCandidate(sch=sched, args_info=[])
        candidates.append(candidate)
        res = ms.runner.RunnerResult([runtime], "", 0.0)
        results.append(res)
    return tasks, candidates, results




def update_cost_model(cost_model, samples):
    tasks, all_candidates, all_results = generate_tasks(samples)
    for i in range(len(tasks)):
        context = tasks[i]
        candidates = [all_candidates[i]]
        results = [all_results[i]]
        cost_model.update(context, candidates, results)


def test_cost_model(cost_model, samples):
    tasks, _, _ = generate_tasks(samples)
    predictions = []
    expected = []
    for i in range(len(tasks)):
        tune_ctx = tasks[i]
        print("tune_ctx", tune_ctx, dir(tune_ctx))
        sched = tvm.tir.Schedule(tune_ctx.mod)
        print("sched", sched)
        dummy_candidate = ms.MeasureCandidate(sch=sched, args_info=[])
        print("dummy_candidate", dummy_candidate)
        if True:
            extractor = ms.feature_extractor.PerStoreFeature()
            (dummy_feature,) = extractor.extract_from(
                tune_ctx,
                candidates=[dummy_candidate],
            )
            print("dummy_feature", dummy_feature, dir(dummy_feature))
        dummy_predictions = cost_model.predict(tune_ctx, [dummy_candidate])
        predictions.append(dummy_predictions[0])
        _, expected_runtime = samples[i]
        expected.append(expected_runtime)
    print("predictions", predictions)
    print("expected", expected)


def load_tir(tir_path):
    with open(tir_path, "r") as f:
        content = f.read()

    # As tvm.script.from_source is unable to handle multiple Primfunc in a file
    funct = [x for x in content.split("# from tvm.script import tir as T") if x.strip() ]
    # objs = [tvm.script.from_source(tir_source_code) for tir_source_code in funct]
    objs = []
    for tir_source_code in funct:
        if not tir_source_code.strip():
            continue
        try:
            obj = tvm.script.from_source(tir_source_code)
        except:

            print(f"Error parsing TIR source code, trying workaround for T.realize")
            try: # Format T.realize to be compatible with TVM 0.13
                pattern = r"T\.realize\s*\(([^()]*)\)"
                replacement = r'T.realize(\1, "global", True)'
                new_code = re.sub(pattern, replacement, tir_source_code)
                obj = tvm.script.from_source(new_code)
                print("Successfully replaced T.realize with global realization")
            except Exception as e:
                print(f"Error replacing T.realize: {tir_source_code}")
                raise e

            
        if isinstance(obj, tvm.tir.PrimFunc):
            default_name = "main"
            obj = tvm.IRModule({default_name: obj})
            # obj = tvm.IRModule({obj.attrs["global_symbol"]: obj})
            assert isinstance(obj, tvm.IRModule)
        yield obj

    # ret=[]
    # for obj in objs:
    #     if isinstance(obj, tvm.tir.PrimFunc):
    #         default_name = "main"
    #         obj = tvm.IRModule({default_name: obj})
    #     assert isinstance(obj, tvm.IRModule)
    #     ret.append(obj)
    # return ret
def benchmark_mod(ir_module):
    """
    Builds function from IRModule and returns the mean time taken to execute the function in milliseconds.
    """
    func = ir_module["main"]
    test_inputs = []
    func_name = func.attrs["global_symbol"]
    
    # The function's buffer_map holds the key-value pair of handle to Buffer object
    for _, buffer_obj in func.buffer_map.items():
    
        shape = tuple(int(s) for s in buffer_obj.shape)
        dtype = buffer_obj.dtype
        test_inputs.append(tvm.nd.array(np.random.rand(*shape).astype(dtype)))
    
    lib = tvm.build(ir_module, target="llvm")
    f_timer_before = lib.time_evaluator(func_name, tvm.cpu())

    return f_timer_before(*test_inputs).mean * 1000

def generate_samples_from_session(session_path):
    """
    Generates samples from a session path.
    """
    tir_files = glob.glob(os.path.join(session_path, "default.tir*"))
    if not tir_files:
        raise ValueError(f"No TIR files found in the session path: {session_path}")
    mods = []
    for tir_file in tir_files:
        mods.extend(load_tir(tir_file))
    return [(mod, benchmark_mod(mod)) for mod in tqdm(mods)]

def main():
    parser = argparse.ArgumentParser(
        description="Train and/or test a TVM MetaSchedule cost model"
    )

    parser.add_argument(
        "--input-model",
        type=Path,
        help="Path to load an existing cost model file"
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        help="Path to save the trained cost model file"
    )
    parser.add_argument(
        "--session-path",
        type=Path,
        help=(
            "Path to the directory with tir dumps"
        ),
        required=True
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the order of samples before splitting"
    )
    parser.add_argument(
        "--split-samples",
        type=float,
        help=(
            "Fraction of samples to use for training (0.0–1.0). "
            "If not set, the same samples will be used for training and testing."
        )
    )

    args = parser.parse_args()
    cost_model = create_cost_model()
    cost_model.num_warmup_samples = 1  # Do not get random predictions. TODO: expose
    if args.input_model is not None:
        cost_model_file = args.input_model
        logging.info("Reading cost model from disk... (%s)", cost_model_file)
        cost_model.load(str(cost_model_file))
    if args.samples is not None and len(args.samples) > 0:
        logging.info("Processing Samples")
        samples = args.samples
        assert len(samples) % 2 == 0
        samples = [(samples[2*i], samples[2*i+1]) for i in range(len(samples) // 2)]
        samples_cnt = len(samples)
        print("samples", samples)
        samples = [(load_tir(x[0]), float(x[1])) for x in samples] # List of tuples of IR module and run times
    # task_name = mod.func_name
        if args.randomize:
            raise NotImplementedError("randomize sample order")
        if args.split_samples is not None:
            split = args.split_samples
            assert isinstance(split, float)
            train_cnt = int(samples_cnt * split)
            assert 0 <= train_cnt <= samples_cnt
            test_cnt = samples_cnt - train_cnt
            assert 0 <= test_cnt <= samples_cnt
            train_samples = samples[:train_cnt]
            assert len(train_samples) == train_cnt
            test_samples = samples[train_cnt:]
            assert len(test_samples) == test_cnt
        else:
            train_samples = test_samples = samples
            train_cnt = test_cnt = samples_cnt
        logging.info("sample_cnt: %d, train_cnt: %d, test_cnt: %d", samples_cnt, train_cnt, test_cnt)
    else:
        logging.info("No samples provided")
        train_samples = []
        test_samples = []
    if len(train_samples) > 0:
        logging.info("Training cost model")
        update_cost_model(cost_model, train_samples)
    else:
        logging.info("Skipping training (train_set empty)")
    if args.output_model is not None:
        cost_model_file = args.output_model
        logging.info("Writing cost model to disk... (%s)", cost_model_file)
        cost_model.save(str(cost_model_file))
    if len(test_samples) > 0:
        logging.info("Testing cost model")
        test_cost_model(cost_model, test_samples)
    else:
        logging.info("Skipping testing (test_set empty)")


if __name__ == "__main__":
    main()
