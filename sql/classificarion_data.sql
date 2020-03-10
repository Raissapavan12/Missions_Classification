SELECT 
 activityId
, date
, workerId
, companyName
, establishmentName
, numberOfRuptures
, numberProductsOutStorage
, numberProductsOutShelf
, rateOfRuptures
, rateOutStorage
, mixSize
, duration
, numberOfPictures
, averageRating
, CASE WHEN averageRating <= 2 THEN 1 ELSE 0 END AS marker
FROM `anthor-prod.DW2.vw_activities_details`
    --WHERE averageRating IS NULL 
    WHERE averageRating IS NOT NULL 
    AND missionType IS NULL
    AND companyName <> 'CONDOR - NILO PEÇANHA' 
    AND companyName <> 'CONDOR - ÁGUA VERDE' 
    AND companyName <> 'CONDOR - CHAMPAGNAT'
    AND companyName <> 'Condor - Piraquara'
    AND companyName <> 'COCA-COLA FEMSA'
    AND status = 'finished'
    AND date >= '2019-08-01'
    AND RAND() < 10/75