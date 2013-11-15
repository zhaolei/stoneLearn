<?php
$a = file('t0');

foreach($a as $t) {
    $x = explode(',', $t);
    foreach($x as $w) {
        $w = intval($w);
    
        //if($w == 33) echo $w;
        $r[$w] = empty($r[$w]) ? 1 : $r[$w]+1; 
    }

}
foreach($r as $k=>$v) {
    echo "$v $k\r\n";
}

//print_r($r);
//sort($r);
//print_r($r);
