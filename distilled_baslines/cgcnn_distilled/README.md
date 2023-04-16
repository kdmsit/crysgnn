# Distilled CGCNN

This repor contains the Distilled version of [CGCNN](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) model. 
We have taken the code from [cgcnn](https://github.com/txie-93/cgcnn) github repo and include "crysgnn" directory into it and do minor changes to retrofit our CrysGNN through Knowledge Distillation loss.
To train the model, please follow the following command

```bash
python train.py --epochs 1000 --data-path '../processed_data/<dataset>/<property>/' 
--lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
where 
- dataset can be "jarvis" or "mp_2018"
- property can be name of properties. Refer to "run_prop.sh" for all training scripts.

We processed data of MP-2018 (2 Properties) and Jarvis (19 Properties) dataset for faster training and put into "../processed_data/" directory.
##  Results

#### Materials Project Dataset (MP 2018.6.1)
- Train data size : 60000 
- Validation data size : 5000 
- Test data size: 4239
<table>
  <tr>
    <th>Property</th>
    <th>CGCNN</th>
    <th>distilled_CGCNN</th>
  </tr>
  <tr>
    <td>formation energy per atom</td>
    <td> 0.039 </td>
    <td>0.032</td>
  </tr>
  <tr>
    <td>band gap</td>
    <td> 0.388 </td>
    <td>0.293</td>
  </tr>
</table>

#### JARVIS DFT Dataset
- Train data size : 80%
- Validation data size : 10%
- Test data size: 10%
<table>
  <tr>
    <th>Property</th>
    <th>CGCNN</th>
    <th>distilled_CGCNN</th>
  </tr>
  <tr>
    <td>formation energy per atom</td>
    <td> 0.063 </td>
    <td> 0.047 </td>
  </tr>
  <tr>
    <td>band gap (OPT)</td>
    <td> 0.200 </td>
    <td> 0.160 </td>
  </tr>
</table>
