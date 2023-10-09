apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: {{WORKER_NAME}}
spec:
  minAvailable: {{MIN_NUM_WORKERS}}
  schedulerName: volcano
  queue: {{QUEUE}}
  plugins:
    env: []
    svc: []
  policies:
    - event: PodEvicted
      action: RestartJob
  tasks:
    - replicas: {{MAX_NUM_WORKERS}}
      name: worker
      policies:
        - event: TaskCompleted
          action: CompleteJob
      template:
        spec:
          imagePullSecrets:
            - name: default-secret
          containers:
            - name: tensorflow
              #image: tensorflow/tensorflow:2.12.0-gpu
              image: ghcr.io/rmrschub/tensorflow-2.13.0-gpu:latest
              imagePullPolicy: Always
              env:
                - name: HOME
                  value: /home/jovyan # <-- HOME is /root by default, but we should use the same home directory as in the codeserver image
                - name: MODEL_PATH
                  value: ./keras-model # /tmp/keras-model
                - name: TEMP_MODEL_PATH
                  value: ./keras-model-temp # /tmp/keras-model # /var/data/keras-model
                # DELETE_TEMP_MODEL_PATH = 1 sometimes will cause following error
                # terminate called without an active exception
                # Aborted (core dumped)
                # https://github.com/tensorflow/tensorflow/issues/50853
                # https://github.com/tensorflow/tensorflow/issues/55250
                - name: DELETE_TEMP_MODEL_PATH
                  value: "1"
              command:
                - sh
                - -c
                - |
                  whoami
                  nvidia-smi --format=csv,noheader --query-gpu=name,serial,uuid,pci.bus_id;
                  WORKER_HOST=`cat /etc/volcano/worker.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  export TF_CONFIG="{\"cluster\":{\"worker\":[${WORKER_HOST}]},\"task\":{\"type\":\"worker\",\"index\":${VK_TASK_INDEX}}}";
                  echo "TF_CONFIG=";
                  echo "$TF_CONFIG" | python3 -m json.tool;
                  ulimit -c unlimited;
                  python3 src/train.py;
                  ERR=$?;
                  echo "Error: $ERR";
                  if [ $ERR -eq 134 ]; then
                    ls -lah;
                    echo "Process PID: $$";
                    while true; do sleep 10; done;
                  fi
#                  if [ $VK_TASK_INDEX -eq 0 ]; then
#                    if [ -e /tmp/keras-model ]; then
#                      rm -rf /tf/keras-model;
#                      mv -v /tmp/keras-model /tf;
#                    fi;
#                  fi;
              ports:
                - containerPort: 2222
                  name: tfjob-port
              resources:
                limits:
                  nvidia.com/gpu: {{NUM_GPUS_PER_WORKER}} # <-- number of GPUs per pod
              workingDir: /home/jovyan/{{WORKING_DIR}} # <-- Working directory where classification.py is located
              securityContext: # <--- Use this to run the container as a specific user instead of root
                allowPrivilegeEscalation: false
                runAsUser: 1000 # jovyan user ID as in codeserver
                runAsGroup: 100 # jovyan user group ID as in code server
              volumeMounts:
                - name: workplace-volume
                  mountPath: /home/jovyan # <-- same mount path as in the codeserver deployment file
                - name: tfds-cache-volume
                  mountPath: /tensorflow_datasets
                  
          volumes:
            - name: tfds-cache-volume  # creates a TFDS cache volume in RAM
              emptyDir: 
                medium: Memory
            - name: workplace-volume
              persistentVolumeClaim:
                claimName: {{WORKPLACE_PVC}} # <-- same PVC name as in the codeserver deployment file
          restartPolicy: Never