-- DROP SCHEMA public CASCADE;
-- CREATE SCHEMA public;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS ltree;

CREATE TABLE IF NOT EXISTS record (
	id bigint primary key,
	factors VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS color (
	id bigint primary key,
	name varchar(255) unique,
	factors VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS pattern (
	id bigint primary key,
	name varchar(255) unique,
	factors VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS garment (
	id bigint primary key,
	name varchar(255) unique,
	factors VECTOR(1536),
	path ltree
);
