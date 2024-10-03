-- 데이터베이스의 타임존을 'Asia/Seoul'로 설정 -> 영구적으로 
ALTER DATABASE service_survey_db SET timezone TO 'Asia/Seoul';

-- created_at 속성에 대한 값을 예시로 들어서 2023-12-06 17:51:03으로 저장한다. 
ALTER TABLE service_survey 
ALTER COLUMN created_at SET DEFAULT date_trunc('second', CURRENT_TIMESTAMP);