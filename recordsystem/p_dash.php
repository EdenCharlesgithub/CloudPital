<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Patient Register</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <link rel="stylesheet" href="../css/style.css">

    <style>

body {
            background-image: url("https://i.ibb.co/YQXGB0L/background-1.png");
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
   
        }
     .card {
            background: rgba(255, 255, 255, 0.85);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin-bottom: 20px;
            display:block;
        }

        .card-title {
            color: #000;
        }

        .card-text {
            color: #000;
        }

    </style>

  </head>
  <body>
  <?php
session_start();
require_once 'conn.php';

if(!isset($_SESSION['contact'])){
    header("Location: p_login.php");
    exit();
}

$contact = $_SESSION['contact'];

$query = "SELECT * FROM users WHERE contact = '$contact'";
$result = mysqli_query($conn, $query);
$row = mysqli_fetch_assoc($result);

if(!$row){
    header("Location: p_login.php");
    exit();
}

$height = $row['height']; // Retrieve from database
$weight = $row['weight']; // Retrieve from database

if ($height > 0 && $weight > 0) {
    $bmi = $weight / pow($height, 2);
} else {
    $bmi = 0;
}

?>

<nav class="navbar bg-body-tertiary sticky-top" style="background: rgba( 255, 255, 255, 0.25 );
          box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
          backdrop-filter: blur( 4px );
          -webkit-backdrop-filter: blur( 4px );
          border-radius: 10px;
          border: 1px solid rgba( 255, 255, 255, 0.18 );">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">
                    <img src="https://i.ibb.co/4RVT7kB/acube-traders-1.png" alt="Bootstrap" width="40" height="40">
                    CloudPital
                  </a>
              <form class="d-flex" role="search">
                <a href="http://127.0.0.1:5000/"><button type="button" class="btn btn-success">CheckUp</button></a>
                <a href="logout.php"><button type="button" class="btn btn-danger">Logout</button></a>
              </form>
            </div>
          </nav>


    
<div class="container mt-5">
    <div class="row row-cols-1 row-cols-md-1 g-4">
        <!-- Card 1 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Symptoms</h5>
                    <p class="card-text"><?php echo $row['symptoms']; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 2 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Blood Group</h5>
                    <p class="card-text"><?php echo $row['bloodgrp']; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 3 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Sugar Level</h5>
                    <p class="card-text"><?php echo $row['sugarlvl']."mmol/L";?></p>
                </div>
            </div>
        </div>

        <!-- Card 4 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Blood Pressure</h5>
                    <p class="card-text"><?php echo $row['bloodp']."mmHg"; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 5 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Other Details</h5>
                    <p class="card-text"><?php echo $row['otherdetails']; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 6 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Doctor Assigned</h5>
                    <p class="card-text"><?php echo $row['doctorassign']; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 7 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Height</h5>
                    <p class="card-text"><?php echo $row['height']."M"; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 8 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Weight</h5>
                    <p class="card-text"><?php echo $row['weight']."Kg"; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 9 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Age</h5>
                    <p class="card-text"><?php echo $row['age']."Years"; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 10 -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Gender</h5>
                    <p class="card-text"><?php echo $row['gender']; ?></p>
                </div>
            </div>
        </div>

        <!-- Card 11 (BMI) -->
        <div class="col">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">BMI</h5>
                    <p class="card-text"><?php echo $bmi; ?></p>
                </div>
            </div>
        </div>

    </div>
</div>


          <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
  </body>
</html>