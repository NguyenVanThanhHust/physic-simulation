# physic-simulation

## Install 
Build docker image
```
docker build -t physic_sim_img -f Dockerfile .
```

Start docker container
```
docker run --rm -it --name physic_sim_ctn --gpus=all --shm-size 8G --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e CUDA_CACHE_DISABLE=0 --env="QT_X11_NO_MITSHM=1" --volume="$PWD:/workspace/" -w /workspace/ physic_sim_img /bin/bash
```

