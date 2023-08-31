with open('Benchmark/Data/nasbench/nasbench/lib/training_time.py', 'r') as f:
    code = f.read()
    
with open('Benchmark/Data/nasbench/nasbench/lib/training_time.py', 'w') as f:
    code = code.replace('tf.train.SessionRunHook', 'tf.estimator.SessionRunHook')
    code = code.replace('tf.train.CheckpointSaverListener', 'tf.compat.v1.train.CheckpointSaverListener')
    f.write(code)

with open('Benchmark/Data/nasbench/nasbench/lib/evaluate.py', 'r') as f:
    code = f.read()
    
with open('Benchmark/Data/nasbench/nasbench/lib/evaluate.py', 'w') as f:
    code = code.replace('tf.train.NanLossDuringTrainingError', 'tf.compat.v1.train.NanLossDuringTrainingError')
    f.write(code)

with open('Benchmark/Data/nasbench/nasbench/api.py', 'r') as f:
    code = f.read()
    
with open('Benchmark/Data/nasbench/nasbench/api.py', 'w') as f:
    code = code.replace('tf.python_io.tf_record_iterator', 'tf.compat.v1.python_io.tf_record_iterator')
    f.write(code)


with open('Benchmark/Data/nasbench/nasbench/scripts/generate_graphs.py', 'r') as f:
    code = f.read()
    
with open('Benchmark/Data/nasbench/nasbench/scripts/generate_graphs.py', 'w') as f:
    code = code.replace('tf.gfile.Open', 'open')
    f.write(code)

