<?php
$a = file('t1');

$arr=array();

foreach($a as $t) {
    $t = intval($t);
    $arr[$t]++;
}

$c = count($arr);
$x = 0.0;
foreach($arr as $k=>$v) {
    $w = $v/$c;
    $f1 = log($w, 2);
    $m = $w * $f1;

    $x += $m * -1;
}

echo $x;
