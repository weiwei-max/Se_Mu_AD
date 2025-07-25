export CARLA_ROOT=${1:-/home/wei/dw/carla_garage/carla}
export WORK_DIR=${2:-/media/wei/Data/code/carla_garage}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/lav.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/pre_train/pretrained_models/lav/tfpp_02_05_withheld_0/transfuser_plus_plus_lav.json
export TEAM_AGENT=${WORK_DIR}/team_code/sensor_agent.py
export TEAM_CONFIG=${WORK_DIR}/pre_train/pretrained_models/lav/tfpp_02_05_withheld_0
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export SAVE_PATH=${WORK_DIR}/results
export UNCERTAINTY_THRESHOLD=0.33
export BENCHMARK=longest6


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--resume=${RESUME} \
--timeout=600
