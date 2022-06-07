echo "This takes about 10 minutes and creates about 5GB of files"

# preliminary datasets
python3.9 PAN12/create_datapack.py --datapackID PAN12
#python ChatCoder2/create_datapack.py --datapackID ChatCoder2

# actually used datasets
python3.9 VTPAN/create_datapack.py --datapackID VTPAN --PAN12datapackID PAN12
#python PANC/create_datapack.py --datapackID PANC --PAN12datapackID PAN12 --CC2datapackID ChatCoder2

# .tsv files for compatibility
python3.9 create_csv.py --dataset VTPAN
#python create_csv.py --dataset PANC
