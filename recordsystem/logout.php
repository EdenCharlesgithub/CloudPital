<?php
session_start(); // Start session

// Unset all session variables
$_SESSION = array();

// Destroy the session
session_destroy();

// Redirect to the specified URL
header("Location: http://127.0.0.1:5000/");
exit();
?>
