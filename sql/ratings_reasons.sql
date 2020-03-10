--Ratings Opened--
SELECT
_id
, code
, score
, activityId
, createdAt
, _id as workerId
, name
, observation
FROM
(
SELECT
rt._id
, SPLIT(REPLACE(REGEXP_REPLACE(reasons, '[^0-9,a-z]',''), 'oid', ''),',') as codeReason
, score
, activityId
, rt.createdAt
, w._id as workerId
, w.name
, observation
FROM `anthor-prod.DW.ratings` rt
LEFT JOIN `anthor-prod.DW.activities` a ON CONCAT('ObjectId(', rt.activityId, ')') = a._id 
LEFT JOIN `anthor-prod.DW.workers` w ON CONCAT('ObjectId(', a.workerId, ')') = w._id
ORDER BY score DESC
),
UNNEST(codeReason) as code

--Ratings Reasons--
SELECT 
_id
, string_field_1 as reasonId
, string_field_2 as reason
, code
, score
, activityId
, createdAt
, workerId
, name
, observation
FROM `anthor-prod.DW.ratingreasons` rr
JOIN `anthor-prod.DW.vw_ratings_opened` ro ON ro.code = REPLACE(REGEXP_REPLACE(rr.string_field_1, '[^0-9,a-z]',''), 'bjectd', '')
WHERE _id IS NOT NULL

--Top Reasons--
SELECT 
COUNT(DISTINCT activityId) as activities
,AVG(score) as score
,COUNT(DISTINCT name) as workers
,reason
FROM DW.vw_ratings_reasons
GROUP BY reason
ORDER BY 1

--Alpha
SELECT 
COUNT(e._id)
,a.status
FROM `anthor-prod.DW.events` e
LEFT JOIN DW.users u ON CONCAT('ObjectId(', e.userId, ')') = u._id
LEFT JOIN DW.activities a ON CONCAT('ObjectId(', e.entityId, ')') =  a._id
WHERE e.event = 'activity-start' AND e.platform LIKE '%alpha' AND LENGTH(platform) >= 21
GROUP BY a.status

--Produtos Smart
SELECT
  DISTINCT(productId)
  ,p.name
  ,currentStock
  ,currentVariation
  ,visitDate
FROM `anthor-prod.DW.activitydynamicproducts` dp
LEFT JOIN DW.products p ON CONCAT('ObjectId(', dp.productId, ')') = p._id
WHERE currentStock IS NOT NULL AND currentVariation > 0
GROUP BY productId
  ,p.name
  ,currentStock
  ,currentVariation
  ,visitDate
ORDER BY p.name

--Jessica
SELECT
COUNT(_id) as missions,
COUNT(DISTINCT workerId) as workers,
SUM(actualPrice)/100 as paid,
COUNT(DISTINCT establishmentId) as establishments,
COUNT(DISTINCT companyId) as companies,
 ROUND(
      AVG
        (
          DATETIME_DIFF(
            DATETIME(TIMESTAMP(LTRIM(RTRIM(JSON_EXTRACT(a.checkedOutAt, '$.date'), '"'), '"'))), 
            DATETIME(TIMESTAMP(LTRIM(RTRIM(JSON_EXTRACT(a.checkedInAt, '$.date'), '"'), '"'))), 
            HOUR
          )
         ),
      2) AS time
FROM `anthor-prod.DW.activities` a
WHERE formattedDate >= '2019-01-01' AND status = 'finished'