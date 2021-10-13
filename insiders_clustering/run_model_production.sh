# variable
data=$(date +'%Y-%m-%dT%H:%M:%S')

#_ path 
path='/home/ubuntu/pa005_insiders_clustering/insiders_clustering'
path_to_envs='/home/ubuntu/.local/bin/'

$path_to_envs/papermill $path/src/models/c10-mdfl-deploy.ipynb $path/reports/c109-mdfl-deploy-$data.ipynb
