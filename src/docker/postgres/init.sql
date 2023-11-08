-- DROP SCHEMA public CASCADE;
-- CREATE SCHEMA public;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS record (
	id bigint primary key,
	factors VECTOR(1536)
);
