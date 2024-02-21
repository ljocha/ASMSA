#!/bin/bash

# XXX: must be here
img=$(cat IMAGE)

# XXX: cwd root of the volue only

if [ -z "$DOCKER_WORKDIR" ]; then
	pvcid=$(df . | tail -1 | cut -f1 -d ' ' | sed 's/^.*pvc-/pvc-/')
	PVC=$(kubectl get pvc | grep $pvcid | cut -f1 -d' ')

	kubectl get job/asmsa-gmx 2>/dev/null || kubectl apply -f - <<EOF
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

	kubectl wait --for=condition=ready pod -l job=asmsa-gmx
	kubectl exec -ti job/asmsa-gmx -- gmx "$@"
else
	docker run -ti -v $DOCKER_WORKDIR:/work -w /work -u $(id -u) $img gmx "$@"
fi

