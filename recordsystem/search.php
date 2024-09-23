<?php
include('conn.php');

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $searchTerm = mysqli_real_escape_string($conn, $_POST['searchTerm']);

    $query = "SELECT * FROM users WHERE contact LIKE '%$searchTerm%' OR name LIKE '%$searchTerm%'";
    $result = mysqli_query($conn, $query);

    if (mysqli_num_rows($result) > 0) {
        while ($row = mysqli_fetch_assoc($result)) {
            echo '<div class="userResult">';
            echo '<h3>' . $row['name'] . '</h3>';
            echo '<p>Contact: ' . $row['contact'] . '</p>';
            echo '<p>Age: ' . $row['age'] . '</p>';
            echo '<p>Gender: ' . $row['gender'] . '</p>';
            echo '</div>';
        }
    } else {
        echo '<p>No results found.</p>';
    }
}
?>
