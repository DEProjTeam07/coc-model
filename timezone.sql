-- PostgreDB 타임존을 UTC로 한다. 
ALTER DATABASE postgres SET TIMEZONE='UTC';


-- created_at 속성에 대한 값을 예시로 들어서 2023-12-06 17:51:03으로 저장한다. 
ALTER TABLE service_survey 
ALTER COLUMN created_at SET DEFAULT date_trunc('second', CURRENT_TIMESTAMP);