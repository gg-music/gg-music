docker container rm -f gtzan_group
docker run -d \
	--gpus all \
	-p 23333:22 \
	--shm-size=4gb \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v /etc/group:/etc/group:ro \
	-v /mnt/4T_HGST/aia-group:/home/gtzan \
	-v /mnt/1T_SSD/aia-group:/home/gtzan/ssd \
	--name gtzan_group gtzan_ssh 
