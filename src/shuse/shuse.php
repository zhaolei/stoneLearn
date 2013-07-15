<?php
$arrS = file('shuse.txt');

foreach($arrS as $t) {
	$t = iconv('gbk', 'utf8', $t);
	$j = json_decode($t, true);
	$d = $j['result']['historyCodes'];
	foreach($d as $tt) {
		//print_r($tt);exit;
		echo "$tt[1] # $tt[2] \r\n";
	}
}
