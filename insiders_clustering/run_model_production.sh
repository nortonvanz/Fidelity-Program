# variable
data=$(date +'%Y-%m-%dT%H:%M:%S')

#_ path 
path='/home/ubuntu/pa005_insiders_clustering/insiders_clustering'
path_to_envs='/home/ubuntu/.pyenv/versions/3.8.0/envs/pa005insidersclustering/bin'

$path_to_envs/papermill $path/src/models/look_c0.9-mdfl-deploy.ipynb $path/reports/look_c0.9-mdfl-deploy.ipynb_$data.ipynb
