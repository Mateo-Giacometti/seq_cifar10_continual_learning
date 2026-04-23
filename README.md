# TP1 - Vision Artificial Avanzada

Trabajo practico de `I309 - Vision Artificial Avanzada` (UdeSA) sobre aprendizaje continuo en Seq-CIFAR-10.

Actualmente cubre y refactoriza las etapas:
- `4.1` Preparacion de dataset secuencial + replay buffer.
- `4.2` Pre-entrenamiento SupCon en Task 0 + linear evaluation.
- `4.3.1` Fine-tuning naive.
- `4.3.2` EWC.
- `4.3.3` LwF.
- `4.3.4` Co2L.
- `[Opcional]` ER-ACE.

## Estructura

```
.
├── configs/
│   └── cifar10.yaml
├── src/
│   └── tp1_cl/
│       ├── __init__.py
│       ├── config.py
│       ├── checkpoints.py
│       ├── data.py
│       ├── models.py
│       ├── train.py
│       ├── viz.py
│       └── methods/
│           ├── __init__.py
│           ├── finetuning.py
│           ├── ewc.py
│           ├── lwf.py
│           ├── co2l.py
│           └── er_ace.py
├── tp1.ipynb
├── project/
├── papers/
└── requirements.txt
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

1. Abrir `tp1.ipynb` desde la raiz del repositorio.
2. Ejecutar las celdas en orden.
3. El notebook usa `configs/cifar10.yaml` como fuente unica de hiperparametros.
4. Protocolo activo: `from_task0_pretrained`.
   - La Task 0 se entrena en `4.2`.
   - Los metodos de CL (`4.3.x`) arrancan desde Task 1 (`task_ids=[1,2,3,4]`) reutilizando estado inicial de Task 0.

## Checkpointing

Se guardan artefactos en `outputs/checkpoints/` para evitar reentrenar cuando solo cambian plots o analisis:
- `task0_supcon.pt`
- `task0_linear_eval.pt`
- `naive_seq_cifar10.pt`
- `ewc_seq_cifar10.pt`
- `lwf_seq_cifar10.pt`
- `co2l_seq_cifar10.pt`
- `er_ace_seq_cifar10.pt`

Las figuras se guardan en `outputs/imgs/` (ademas de mostrarse en notebook):
- `supcon_loss.png`
- `embedding_snapshots.png`
- `linear_eval_accuracy.png`
- `naive_cl_metrics.png`
- `ewc_cl_metrics.png`
- `lwf_cl_metrics.png`
- `co2l_cl_metrics.png`
- `er_ace_cl_metrics.png`
- `methods_over_tasks.png`
- `forgetting_class_il.png`
- `forgetting_task_il.png`
- `bwt_class_il.png`
- `bwt_task_il.png`
- `forgetting_by_task_class_il_<metodo>.png`
- `forgetting_by_task_task_il_<metodo>.png`
- `methods_comparison.png`

Adicionalmente se exporta una tabla comparativa en:
- `outputs/metrics/methods_comparison.csv`

## Metricas de transferencia y olvido

En la seccion 4.4 se calculan metricas sobre la matriz task-wise `A[t, k]`
(accuracy en tarea `k` luego de aprender hasta el paso `t`):

- `Forgetting (max-history)` (recomendado): `F[t, k] = max_{l<t} A[l, k] - A[t, k]` para `k < t`.
- `BWT`: `BWT[t, k] = A[t, k] - A[t0(k), k]`, donde `t0(k)` es el primer paso en que aparece la tarea `k`.

El promedio por paso se calcula solo sobre tareas pasadas (`k < t`), evitando incluir
la tarea actual en el promedio.

Nota sobre matrices task-wise:
- `taskwise_class_il_matrix`: evalua por tarea, pero con decision global Class-IL (argmax sobre todos los logits).
- `taskwise_task_il_matrix`: evalua por tarea con decision restringida a las clases de la tarea (Task-IL).

Si el checkpoint existe, el notebook carga; si no existe, entrena y guarda automaticamente.
Si existe pero no cumple el protocolo `from_task0_pretrained`, se re-entrena y se sobre-escribe.

## Notas de reproducibilidad

- El split por defecto es Seq-CIFAR-10 con 5 tareas x 2 clases.
- Seed global configurable desde `configs/cifar10.yaml`.
- `data/` y `outputs/` no se versionan.
