<?php
deviceSet('COM2');
$serial->deviceOpen();
$serial->sendMessage( ' Sending a message to the port! ' );
$serial->deviceClose();
