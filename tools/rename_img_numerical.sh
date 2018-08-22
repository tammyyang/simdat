a=1
for img in *.jpg; do
    new=$(printf "%04d.jpg" "$a") #04 pad to length of 4
    mv -i -- "$img" "$new"
    echo $img $new
    let a=a+1
done
