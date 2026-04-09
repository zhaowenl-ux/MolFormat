CREATE TABLE IF NOT EXISTS chemistry.compound_70_sorted
(
    compound_id integer NOT NULL,
    x character varying(256) COLLATE pg_catalog."default" NOT NULL,
    y character varying(256) COLLATE pg_catalog."default" NOT NULL,
    bond character varying(512) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT compound_70_sorted_pkey PRIMARY KEY (compound_id)
);


CREATE TABLE IF NOT EXISTS chemistry.logp
(
    logp double precision NOT NULL,
    smiles character varying(256) COLLATE pg_catalog."default" NOT NULL,
    x character varying(512) COLLATE pg_catalog."default",
    y character varying(512) COLLATE pg_catalog."default",
    bond character varying(784) COLLATE pg_catalog."default",
    num_atoms integer,
    id integer NOT NULL DEFAULT nextval('logp_id_seq'::regclass),
    CONSTRAINT logp_pkey PRIMARY KEY (id)
);

