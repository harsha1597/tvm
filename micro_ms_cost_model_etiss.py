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
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Callable
import pickle
import numpy as np

import tvm
import tvm.testing
from tvm import te, tir
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib import utils
from tvm import meta_schedule as ms
from tvm.driver import tvmc
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig
from tvm.meta_schedule.logging import get_logger
from tvm import transform
from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro
from tvm.contrib.micro.meta_schedule.rpc_runner_micro_mem import get_rpc_runner_micro_mem

from tvm.rpc import connect_tracker
from model_info import get_model_info
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.DEBUG)
get_logger("xgb_model").setLevel(logging.DEBUG)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR.parent

GCC_PREFIX = os.environ.get("GCC_PREFIX", "/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/deps/install/riscv_gcc_rv32")
GCC_NAME = os.environ.get("GCC_NAME", "riscv32-unknown-elf")
LLVM_DIR = os.environ.get("LLVM_DIR", "/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/deps/install/llvm")
ETISS_TEMPLATE =  "/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/deps/src/microtvm-etiss-template"
ETISS_SCRIPT = os.environ.get("ETISS_SCRIPT", "/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/deps/install/etiss/bin/run_helper.sh")
PLATFORM = os.path.join(ETISS_TEMPLATE, "template_project")

import sys

def load_model(model):
    def _load_model(path, shape_dict):
        model = tvmc.load(
            str(path),
            shape_dict=shape_dict,
        )
        mod = model.mod
        params = model.params
        return mod, params

    model_info = get_model_info(model)
    shape_dict = {t.name: t.shape for t in model_info.in_tensors}
    assert len(model_info.in_tensors) == 1
    input_name, input_shape = list(shape_dict.items())[0]
    input_dtype = model_info.in_tensors[0].dtype
    data_sample = np.random.rand(*input_shape).astype(input_dtype)
    mod, params = _load_model(model, shape_dict)
    return mod, params, input_name, input_shape, input_dtype, data_sample


def get_tuning_config():
    def _get_sch_rules():
        structure = "SR"
        return [
            ms.schedule_rule.ApplyCustomRule(),
            ms.schedule_rule.InlineConstantScalars(),
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            ms.schedule_rule.MultiLevelTiling(
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 2, 4, 8, 16, 32, 64],
                unroll_explicit=True,
                # unroll_explicit=False,
            ),
            ms.schedule_rule.RandomComputeLocation(),
        ]

    def _get_postprocs():
        return [
            ms.postproc.DisallowDynamicLoop(),
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
        ]

    def _get_mutator_probs():
        return {
            ms.mutator.MutateTileSize(): 0.9,
            ms.mutator.MutateComputeLocation(): 0.05,
            ms.mutator.MutateUnroll(): 0.03,
            # ms.mutator.Parallel(): 0.02,
        }

    sch_rules = _get_sch_rules()
    postprocs = _get_postprocs()
    mutator_probs = _get_mutator_probs()
    return sch_rules, postprocs, mutator_probs


def _schedule_dummy():

    def schedule_fn(sch, block=None) -> bool:
        return True

    return schedule_fn



def test_micro_tuning_with_meta_schedule(platform, opt_params, target, num_trials_per_iter, max_trials_per_task, max_trials_global, module_equality, model, transform_layout, options, task_filter):
    
    (opt_level, pass_config, disabled_pass) = opt_params 

    KEEP = True
    if KEEP:
        base_dir = Path("./tune_logs/")
        now = datetime.now()
        ts = now.strftime("%Y%m%dT%H%M%S")

        label = ts
        work_dir_path = base_dir / label
    else:
        work_dir = utils.tempdir()
        work_dir_path = work_dir.path
    print("work_dir_path", work_dir_path)
    
    mod, params, input_name, input_shape, input_dtype, data_sample = load_model(model)

    if transform_layout:
        with tvm.transform.PassContext(
            opt_level=opt_level,
            config=pass_config,
            disabled_pass=disabled_pass,
        ):
            desired_layouts = {"qnn.conv2d": ["NCHW", "default"]}

            # Convert the layout of the graph where possible.
            seq = transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                    relay.transform.FoldConstant(),
                ]
            )
            mod = seq(mod)

    link_params = True

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": link_params})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    builder = get_local_builder_micro()

    with ms.Profiler() as profiler:
        if not SKIP_TUNING:
            sch_rules, postprocs, mutator_probs = get_tuning_config()
            space = ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs=mutator_probs,
            )
            strategy = "evolutionary"
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            extractor = ms.feature_extractor.PerStoreFeature()
            num_warmup_samples = 10
            #cost_model = ms.cost_model.XGBModel(extractor=extractor, num_warmup_samples=num_warmup_samples)
            cost_model = ms.cost_model.RandomModel()
            # micro_rpc_workers = num_trials_per_iter
            with get_rpc_runner_micro(
                platform=platform, options=options, session_timeout_sec=120, evaluator_config=evaluator_config,
                # serial_numbers=["micro"] * micro_rpc_workers,
                tracker_host="127.0.0.1",
                tracker_port=9190,
                # max_workers=micro_rpc_workers,
                rpc_timeout_sec=10,

            ) as runner:
                tracker = connect_tracker("127.0.0.1", 9190)
                print("Tracker summary:\n", tracker.summary(), "\n max_trials_global: ",max_trials_global)

                if max_trials_global > 0:
                    ## tasks are induvidual functions
                    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
                        extracted_tasks=ms.relay_integration.extract_tasks(
                            mod,
                            target,
                            params,
                            opt_level=opt_level,
                            module_equality=module_equality,
                            pass_config=pass_config,
                            disabled_pass=disabled_pass,
                        ),
                        work_dir=str(work_dir_path),
                        space=space,
                        strategy=strategy,
                        num_tuning_cores=1,
                    )
                    if task_filter is not None:
                        assert isinstance(task_filter, list)
                        assert len(task_filter) > 0
                        tasks = [tasks[i] for i in task_filter]
                        task_weights = [task_weights[i] for i in task_filter]
                    pass_config = dict(pass_config)
                    with transform.PassContext(
                        opt_level=opt_level,
                        config=pass_config,
                        disabled_pass=disabled_pass,
                    ):
                        db: ms.Database = ms.tune.tune_tasks(
                            tasks=tasks,
                            task_weights=task_weights,
                            work_dir=str(work_dir_path),
                            max_trials_global=max_trials_global,
                            max_trials_per_task=max_trials_per_task,
                            num_trials_per_iter=num_trials_per_iter,
                            builder=builder,
                            runner=runner,
                            cost_model=cost_model,
                            module_equality=module_equality,
                        )
                else:
                    print("Failed, _schedule_dummy")
                    # db = ms.database.MemoryDatabase()
                    db = ms.database.ScheduleFnDatabase(
                        _schedule_dummy()
                    )
    with open(work_dir_path / "options.json", "w") as f:
        f.write(str(options))
    return db

            #  Build model using meta_schedule logs
    #         ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
    #             database=db,
    #             mod=mod,
    #             target=target,
    #             params=params,
    #             pass_config=MappingProxyType(
    #                 {
    #                     **pass_config,
    #                     "relay.backend.use_meta_schedule": True,
    #                     "relay.backend.tir_converter": "default",
    #                     "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
    #                 }
    #             ),
    #             disabled_pass=disabled_pass,
    #             executor=executor,
    #             runtime=runtime,
    #         )
    # print("tasks[0]", tasks[0], dir(tasks[0]))
    # print("tasks[0]", tasks[0].mod)
    # print(profiler.table())
    # print("cost_model", cost_model, dir(cost_model))
    # saved_model_path = work_dir_path / "cost_model.tar"
    # # random_state = model.extractor.random_state
    # cost_model.save(str(saved_model_path))
    # cost_model.load(str(saved_model_path))
    # cost_model.num_warmup_samples = 1  # Do not get random predictions
    # # model.extractor.random_state = random_state
    # # candidate = MeasureCandidate(Schedule(FullModule), [])
    # dummy_preds = []
    # record_preds = []
    # for i in range(len(tasks)):
    #     tune_ctx = tasks[i]
    #     print("tune_ctx", tune_ctx, dir(tune_ctx))
    #     sched = tir.Schedule(tune_ctx.mod)
    #     print("sched", sched)
    #     # dummy_candidate = _make_candidate(sched)
    #     dummy_candidate = ms.MeasureCandidate(sch=sched, args_info=[])
    #     print("dummy_candidate", dummy_candidate)
    #     (dummy_feature,) = extractor.extract_from(
    #         tune_ctx,
    #         candidates=[dummy_candidate],
    #     )
    #     print("dummy_feature", dummy_feature, dir(dummy_feature))
    #     dummy_predictions = cost_model.predict(tune_ctx, [dummy_candidate])
    #     dummy_preds.append(dummy_predictions[0])
    #     print("dummy_predictions", dummy_predictions)
    #     workload = db.commit_workload(tasks[i].mod)
    #     records = db.get_top_k(workload, 3)
    #     print("records", records, len(records))
    #     if len(records) == 0:
    #         continue
    #     record = records[0]
    #     print("record", record, dir(record))
    #     db.commit_tuning_record(record)
    #     record_trace = record.trace
    #     print("record_trace", record_trace, dir(record_trace))
    #     record_sched = tir.Schedule(record.workload.mod)
    #     print("record_sched_init", record_sched, dir(record_sched))
    #     record_trace.apply_to_schedule(record_sched, remove_postproc=False)
    #     print("record_sched", record_sched, dir(record_sched))
    #     record_candidate = ms.MeasureCandidate(sch=record_sched, args_info=[])
    #     print("record_candidate", record_candidate)
    #     (record_feature,) = extractor.extract_from(
    #         tune_ctx,
    #         candidates=[record_candidate],
    #     )
    #     print("record_feature", record_feature, dir(record_feature))
    #     record_predictions = cost_model.predict(tune_ctx, [record_candidate])
    #     print("record_predictions", record_predictions)
    #     # assert len(record_predictions) == 1
    #     record_preds.append(record_predictions[0])
    # print("dummy_preds", dummy_preds)
    # sorted_dummy_idxs = list(np.argsort(dummy_preds))
    # print("sorted_dummy_idxs", sorted_dummy_idxs)
    # sorted_dummy_preds = [dummy_preds[i] for i in sorted_dummy_idxs]
    # print("sorted_dummy_preds", sorted_dummy_preds)
    # dummy_preds_sum = sum(dummy_preds)
    # print("dummy_preds_sum", dummy_preds_sum)
    # print("record_preds", record_preds)
    # sorted_record_idxs = list(np.argsort(record_preds))
    # print("sorted_record_idxs", sorted_record_idxs)
    # sorted_record_preds = [record_preds[i] for i in sorted_record_idxs]
    # print("sorted_record_preds", sorted_record_preds)
    # record_preds_sum = sum(record_preds)
    # print("record_preds_sum", record_preds_sum)
    # # TODO: weighted sum!
    # input("!!!")
    # non_ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
    #     None,
    #     mod=mod,
    #     target=target,
    #     params=params,
    #     pass_config=MappingProxyType(
    #         {
    #             **pass_config,
    #             "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
    #         }
    #     ),
    #     disabled_pass=disabled_pass,
    #     executor=executor,
    #     runtime=runtime,
    # )

    # if not SKIP_TUNING:
    #     # TUNED
    #     # TODO: wrap in helper
    #     project = tvm.micro.generate_project(
    #         str(platform),
    #         ms_mod,
    #         str(work_dir_path / "project"),
    #         options=options,
    #     )
    #     project.build()
    #     project.flash()
    #     with tvm.micro.Session(project.transport()) as session:
    #         aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    #         result = aot_executor.module.time_evaluator("run", session.device, number=1)()
    #         print("result", result)
    #         print("mean: ", result.mean)

    # # UNTUNED
    # project = tvm.micro.generate_project(
    #     str(platform),
    #     non_ms_mod,
    #     str(work_dir_path / "project2"),
    #     options=options,
    # )
    # project.build()
    # project.flash()
    # with tvm.micro.Session(project.transport()) as session:
    #     aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    #     result2 = aot_executor.module.time_evaluator("run", session.device, number=1)()
    #     print("result2", result2)
    #     print("mean2:", result2.mean)
    # if not SKIP_TUNING:
    #     rel = result.mean / result2.mean
    #     print("rel:  ", rel)

def get_all_tflite_files(directory):
        tflite_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".tflite"):
                    tflite_files.append(os.path.join(root, file))
        return tflite_files

def transform_feats(raw_feats):
    """ As the features have a variable size: (x,164) where x refers to the number of BufferStores in a Primfunc. We need to 
    aggregate the features """
    num_of_rows = raw_feats.shape[0]
    agg = np.concatenate([
                np.sum(raw_feats, axis=0),
                #np.std(raw_feats, axis=0),
                np.array(num_of_rows).reshape((1))
                
            ]) 
    return agg

def filter_feats(X,y):
    """ Remove features with all 0s and low MI and standardize variables
    returns the filtered_training data, mask, scaler to be used on test data """

    # keep columns with at least one nonzero
    mi = mutual_info_regression(X, y, random_state=42)
    mask_nonzero = (X != 0).any(axis=0)

    # Remove features with zero MI
    mask = mi > 0
    final_mask = mask_nonzero & mask
    X_filtered = X[:, final_mask] if isinstance(X, np.ndarray) else X.loc[:, final_mask]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    return X_scaled,final_mask,scaler

def tuninglog_tofeats(tuning_dir):
    # tuning_dir= "/kaggle/input/tunelogs/content/tmplogs"
    directories = os.listdir(tuning_dir)
    paths = [os.path.join(tuning_dir,x) for x in directories]
    target = tvm.target.Target("c")
    #dbs = []
    #samples=[]
    bad_tune=0
    X_train=[]
    y_train=[]
    for path in paths:
        tuning_records = os.path.join(path, "database_tuning_record.json")
        workload_path = os.path.join(path, "database_workload.json")
        db = ms.database.JSONDatabase(path_tuning_record=tuning_records, path_workload=workload_path)
        
        for i,rec in enumerate(db.get_all_tuning_records()):
            
            time=rec.run_secs
            if(len(rec.run_secs) >1) or time[0] > 1000:
                print(i, time, tuning_records )
                bad_tune+=1
                print("Ignoring this tuning record as it is corrupted")
                continue
                #assert False, "Assuming 1 run_sec per record"
                
            y_train.append(float(time[0]))

            mod=rec.workload.mod
            #samples.append((mod,time))
            tune_ctx = ms.tune_context.TuneContext(
                mod=mod,
                target=target,
                # space_generator=space,
                # search_strategy=strategy,
                # task_name=task_name,
                # logger=logger,
                # rand_state=rand_state,
                # num_threads=num_tuning_cores,
            )
            #tasks.append(ctx)
            sched = tvm.tir.Schedule(mod)
            candidate = ms.MeasureCandidate(sch=sched, args_info=[])

            extractor = ms.feature_extractor.PerStoreFeature()
            (dummy_feature,) = extractor.extract_from(
                tune_ctx,
                candidates=[candidate],
            )
            dummy_feature = dummy_feature.numpy()
            X_train.append(transform_feats(dummy_feature))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_filtered,feature_mask,scaler = filter_feats(X_train,y_train)
    print("X_filtered shape", X_filtered.shape, "y_train.shape: ", y_train.shape)
    print("No of bad tuning samples: ", bad_tune)
    pickle_obj = (X_filtered, y_train, feature_mask, scaler)
    now = datetime.now()
    ts = now.strftime("%Y%m%dT%H%M%S")

    label = ts
    with open(f"featureset_{label}.pickle", "wb") as f:
        pickle.dump(pickle_obj, f)

if __name__ == "__main__":
    # MODEL = "/work/git/mlonmcu/mlonmcu/workspace_default/models/resnet/resnet.tflite"
    model_path= "/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models"
    # MODELS = ["/mobilenet_v1_1_0_224_quant/mobilenet_v1_1_0_224_quant.tflite","/lstm2/lstm2.tflite",
    #           "/cifar10/cifar10.tflite",""]
    # tflite_files=[ '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/aww/aww.tflite',
    #    '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/MobileNetV2/MobileNet_V2.tflite',
    #    '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/lstm2/lstm2.tflite',
    #     '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/vww/vww.tflite',
    #      '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/toycar/toycar.tflite',
    #       '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/resnet/resnet.tflite',
    #        '/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/magic_wand/magic_wand.tflite']
    
    

    # Example usage:
    tflite_files = get_all_tflite_files(model_path)
    
    # print(ETISS_TEMPLATE)
    # assert len(sys.argv) == 2, "Usage: micro_ms_cost_model_etiss.py MODEL_PATH"
    # MODEL = sys.argv[1]
    ALTER_OP = True
    TOOLCHAIN = "gcc"
    TARGET = "c -num-cores 1"
    NUM_TRIALS_PER_ITER, MAX_TRIALS_PER_TASK, MAX_TRIALS_GLOBAL = (5, 10, 1000000)
    TASK_FILTER = list(range(5)) # Tune the top 10 highest FLOPs tasks
    MODULE_EQUALITY = "ignore-ndarray"
    TRANSFORM_LAYOUT = False

    OPTIONS = {
        "verbose": True,
        "quiet": True,
        "gcc_prefix": str(GCC_PREFIX),
        "gcc_name": GCC_NAME,
        "llvm_dir": str(LLVM_DIR),
        "etiss_script": str(ETISS_SCRIPT),
        "etiss_args": "",
        "arch": "rv32gc_zicsr_zifencei",
        "abi": "ilp32d",
        "cpu_arch": "RV32IMACFD",
        "cpu_freq": 100000000,
        "toolchain": "llvm",#TOOLCHAIN,
        "opt":2,
        "gc":1,
        "lto":1
    }
    
    MS_DISPATCH = 1  # silent?
    # MS_DISPATCH = 2  # verbose
    # MS_DISPATCH = ?  # error
    SKIP_TUNING = False

    sw_opt=list(range(1,4))
    gc=[0,1]
    lto=[0,1]
    opt_levels = list(range(1, 4))
    max_stack_alloca_vals = [2**k for k in range(1, 12+1)]

    pass_config = {
        "tir.disable_vectorize": True,'tir.max_stack_alloca':1024
    }    
    disabled_pass = ["AlterOpLayout"]
    sys.stdout = open("tune.txt", "w")
    sys.stderr = sys.stdout


    #MODEL = tflite_files[0] #"/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/models/resnet/resnet.tflite"
    for toolchain in ["gcc","llvm"]:
        OPTIONS["toolchain"] = toolchain
        for sw in sw_opt:
            OPTIONS["opt"] = sw
            for g in gc:
                OPTIONS["gc"] = g
                for l in lto:
                    OPTIONS["lto"] = l
                    print("OPTIONS", OPTIONS)
                    for opt in opt_levels[::-1]:
                        for max_stack_alloca in max_stack_alloca_vals:
                            pass_config['tir.max_stack_alloca'] = max_stack_alloca
                            params_config = (opt, pass_config, disabled_pass)
                            
                            for i,MODEL in enumerate(tflite_files[::-1]):
                                print(params_config, MODEL)
                                
                                try:
                                    db = test_micro_tuning_with_meta_schedule(PLATFORM, params_config, TARGET, NUM_TRIALS_PER_ITER, MAX_TRIALS_PER_TASK, MAX_TRIALS_GLOBAL, MODULE_EQUALITY, MODEL, TRANSFORM_LAYOUT, OPTIONS, TASK_FILTER)
                                except Exception as e:
                                    print("Exception:", MODEL, e)
                                    continue
                    # tuning_log_path = "/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/deps/src/tvm/tune_logs"
                    # tuninglog_tofeats(tuning_log_path)
                    # with open("./tir_examples/db.pickle", "wb") as f:
                    #     pickle.dump(db, f)