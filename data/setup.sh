echo "データ拡張中..."
python3 augment.py
echo "特徴量MFCC 作成中..."
python3 mfcc.py Dataset1
python3 mfcc.py Dataset1-a
python3 mfcc.py Dataset2
echo "特徴量Delta 作成中..."
python3 delta.py Dataset1
python3 delta.py Dataset1-a
python3 delta.py Dataset2
echo "特徴量Power 作成中..."
python3 power.py Dataset1
python3 power.py Dataset1-a
python3 power.py Dataset2
