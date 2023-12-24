rm static_type_check.log

for file in ./*.py
do
    echo $file
    mypy $file >> static_type_check.log
done
