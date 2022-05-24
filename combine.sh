tail -n +2 wiki.en.train.csv > en.all.csv
tail -n +2 wiki.en.test.csv >> en.all.csv
tail -n +2 en.train.csv >> en.all.csv
tail -n +2 en.test.csv >> en.all.csv

shuf en.all.csv > en.all.shuf.csv

rm en.all.csv

wc -l en.all.shuf.csv

echo "text,summary" > en.all.train.csv
head -n 900000 en.all.shuf.csv >> en.all.train.csv
echo "text,summary" > en.all.test.csv
tail -n 60000 en.all.shuf.csv >> en.all.test.csv

wc -l en.all.train.csv
wc -l en.all.test.csv
rm en.all.shuf.csv