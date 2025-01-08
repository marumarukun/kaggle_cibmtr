### 01_eda_for_FE.ipynb
- FEのためのEDA
- ほぼ読み込むだけにとどまっている

### 02_gbdt_with_FE.ipynb
- chrisとたかいとさんのnotebookがベース
- 8つのGBDTを作成
- KaplanMeierFitterで目的変数を作成して回帰モデルx3
- Nelson-Aalen推定で目的変数を作成して回帰モデルx3
- COXモデルx2
- 必要最低限のFE
    - 欠損値カウント
    - ドナーと患者の性別一致してるかフラグ
- Nelder-Mead法で重みを最適化
