CREATE DATABASE IF NOT EXISTS GARSDB;
USE GARSDB;

CREATE TABLE IF NOT EXISTS sessions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_image MEDIUMBLOB,
    predicted_age INT,
    predicted_gender VARCHAR(10),
    created_on DATETIME DEFAULT CURRENT_TIMESTAMP
);