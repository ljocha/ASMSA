#!/bin/bash

#echo $0: "$@" >&2
#env >&2

img=cerit.io/ljocha/gromacs:2023-2-plumed-2-9-afed-pytorch-model-cv-2

input=$(cat -)

# XXX: cwd root of the volue only

if [ -z "$DOCKER_WORKDIR" ]; then
	pvcid=$(df . | tail -1 | cut -f1 -d ' ' | sed 's/^.*pvc-/pvc-/')
	PVC=$(kubectl get pvc | grep $pvcid | cut -f1 -d' ')

	mnt=$(df . | tail -1 | cut -f7 -d ' ')
	DIR=$(realpath --relative-base $mnt $PWD)

	kubectl get job/asmsa-gmx 2>/dev/null >/dev/null || kubectl apply -f - >&2 <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: asmsa-gmx
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        job: asmsa-gmx
    spec:
      restartPolicy: Never
      securityContext: # Pod security context
        fsGroupChangePolicy: OnRootMismatch
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: asmsa-gmx
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
            cpu: '.1'
            memory: 8Gi
            nvidia.com/gpu: 0
          limits:
            cpu: '1'
            memory: 8Gi
            nvidia.com/gpu: 0
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

	kubectl wait --for=condition=ready pod -l job=asmsa-gmx >&2
	kubectl exec job/asmsa-gmx -- bash -c "$kcmd <<<\"$input\""
else
	docker run -i --gpus all -v $DOCKER_WORKDIR:/work -w /work -u $(id -u) $img gmx "$@" <<<"$input"
fi

