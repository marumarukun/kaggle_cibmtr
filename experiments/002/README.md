### 01_first_eda.ipynb
- 最低限のEDA（ほぼ読み込むだけ）

### 02_gbdt_target_kaplan_and_cox_model.ipynb
- chrisとたかいとさんのnotebookがベース
- 5つのGBDTを作成
- KaplanMeierFitterで目的変数を作成して回帰モデルx3
- COXモデルx2

### 02_gbdt_target_nelson.ipynb
- 上記notebookに下記追加
- 目的変数をNelson-Aalen推定で作成して回帰モデルx3を追加
- Nelder-Mead法で重みを最適化するコードを追加

### 02_gbdt_target_cox.ipynb
- 上記notebookに下記追加
- 目的変数をCOX推定で作成して回帰モデルx3を追加
- Nelder-Mead法で重みを最適化するコードを追加

### 03_nn_tensorflow.ipynb
- chrisのnotebookそのまま

### 03_nn_pytorch.ipynb
- 上記をpytorchで実装しなおし
- 精度全然出ず、原因不明のまま
- 追記：ターゲット（y）のデータセットクラスに入力する際の形状を適切に修正 → 無事精度出た

### 04_ensamble.ipynb
- Nelder-Mead法で重みを最適化する
