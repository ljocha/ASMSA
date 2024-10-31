#!/bin/bash

#echo $0: "$@" >&2
#env >&2

img=cerit.io/ljocha/gromacs:2023-2-plumed-2-9-afed-pytorch-model-cv-2
: ${REQUEST_CPU:=1}
: ${REQUEST_RAM:=8}
: ${REQUEST_GPU:=0}

limit_cpu=1
limit_ram=8
if [ $REQUEST_CPU -gt $limit_cpu ]; then limit_cpu=$REQUEST_CPU; fi
if [ $REQUEST_RAM -gt $limit_ram ]; then limit_ram=$REQUEST_RAM; fi


input=$(cat -)

# XXX: cwd root of the volume only
gmxjob=asmsa-gmx-$$

if [ -z "$DOCKER_WORKDIR" ]; then
	pvcid=$(df . | tail -1 | cut -f1 -d ' ' | sed 's/^.*pvc-/pvc-/')
	PVC=$(kubectl get pvc | grep $pvcid | cut -f1 -d' ')

	mnt=$(df . | tail -1 | cut -f7 -d ' ')
	DIR=$(realpath --relative-base $mnt $PWD)

	if [ -f $mnt/.gmxjob ]; then 
		gmxjob=$(cat $mnt/.gmxjob)
	else
		echo $gmxjob >$mnt/.gmxjob
	fi

	kubectl get job/$gmxjob 2>/dev/null >/dev/null || kubectl apply -f - >&2 <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $gmxjob
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        job: $gmxjob
    spec:
      restartPolicy: Never
      securityContext: # Pod security context
        fsGroupChangePolicy: OnRootMismatch
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: $gmxjob
        image: $img
        workingDir: /mnt/
        command: [ 'sleep', 'inf' ] 
        securityContext:
          runAsUser: 1000
          runAsGroup: 1000
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
        env:
        - name: 'OMP_NUM_THREADS'
          value: '1'
        resources:
          requests:
            cpu: '$REQUEST_CPU'
            memory: ${REQUEST_RAM}Gi
            nvidia.com/gpu: ${REQUEST_GPU}
          limits:
            cpu: '$limit_cpu'
            memory: ${limit_ram}Gi
            nvidia.com/gpu: ${REQUEST_GPU}
        volumeMounts:
        - name: vol-1
          mountPath: /mnt
      volumes:
      - name: vol-1
        persistentVolumeClaim:
          claimName: '$PVC'
EOF

    kcmd="cd /mnt/$DIR && gmx"
    for c in "$@"; do
        kcmd="$kcmd '$c'"
    done

	kubectl wait --for=condition=ready pod -l job=$gmxjob >&2
	kubectl exec job/$gmxjob -- bash -c "$kcmd <<<\"$input\""
else
	docker run -i --gpus all -v $DOCKER_WORKDIR:/work -w /work -u $(id -u) $img gmx "$@" <<<"$input"
fi

