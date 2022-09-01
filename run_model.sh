# variable
data=$(date +'%Y-%m-%dT%H:%M:%S')


# path (root of project) 
path='/Users/home/repos/pa005_fidelity_program/'

# path where papermill is installed on env (find: 'which papermill' on project root folder)
path_to_envs='/Users/home/opt/anaconda3/envs/pa005_clustering/bin'


$path_to_envs/papermill $path/src/models/9.0-nmv-deploy.ipynb $path/reports/9.0-nmv-deploy-$data.ipynb

