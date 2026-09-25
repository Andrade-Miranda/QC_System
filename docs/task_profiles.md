# Task profiles

Task profiles specify downstream requirements. They determine which evidence domains are task-relevant and which resources must be present; they do not make advisory reasoning authoritative.

| Notation | Repository task mode | Description |
| --- | --- | --- |
| `τP` | `pancreas_only` | Visible pancreas segmentation. |
| `τL` | `pancreas_lesion` | Pancreas lesion segmentation. |
| `τS` | `pancreas_lesion_subregions` | Lesion task with pancreas subregion masks. |

When the task changes, evidence may become task-relevant. Do not interpret one profile as clinically better or universally stricter. The profiles encode different intended downstream segmentation tasks.
