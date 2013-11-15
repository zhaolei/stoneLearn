<?php
$a = file('t0');
$a = array_reverse($a);
$intC = count($a);

$intF = intval($argv[1]);
$intJ = intval($argv[2]);
$intF = empty($intF) ? 1 : $intF;
$intJ = empty($intJ) ? 1 : $intJ;

$s = array();
$ok = array();
$i = 0;
$j = 0;
$sum = 0;
foreach($a as $t) {
    //echo "$t \r\n";
    $x = explode(',', $t);
    
    
            //$sum = 0;
    foreach($x as $w) {
        $w = intval($w);
        if(!empty($s[$i-$intF])) {
            if(in_array($w, $s[$i-$intF])) {
                $ok[$i]++;
            }
        }

        if($j == $intJ)  {
            $j = 0;
            $sum = 0;
        } 
        
        $sum +=$w;
    }

    if(($j+1) != $intJ)  {
        $j++;
        $i++;
        continue; 
    }


    $sum = $sum/(($j+1) * 6);
    //$sum = $sum/6;

    //$s = array();
    $s[$i][1] = $sum * 0.618;
    $s[$i][2] = $sum * 0.382;
    $s[$i][3] = $sum * 0.236;
    
    //$s[$i][1] = intval($s[$i][1]);
    //$s[$i][2] = intval($s[$i][2]);
    //$s[$i][3] = intval($s[$i][3]);
    $s[$i][1] = ceil($s[$i][1]);
    $s[$i][2] = ceil($s[$i][2]);
    $s[$i][3] = ceil($s[$i][3]);
    $s[$i][4] = intval($s[$i][1]) + 1;
    $s[$i][5] = intval($s[$i][2]) + 1;
    $s[$i][6] = intval($s[$i][3]) + 1;

    $j++;
    $i++;
}
$arra = array();
foreach($ok as $m) {
    $arra[$m]++;
}

foreach($arra as $k => $m ) {
    $q = $m/$intC;
    $q = $q * 100;
    echo "$intF $k $q \r\n";
}
echo "-------------------------\r\n";
