/*
===========================================================
Project 5 - Industrial Predictive Maintenance
SQL Server

Includes:
1) Staging table
2) Data validation
3) Relational schema
4) Raw -> cleaned/feature-ready ETL
5) Prediction History
6) 12 analytical queries
7) JOIN, CTE and Window Functions

Dataset: AI4I 2020 Predictive Maintenance
===========================================================
*/

------------------------------------------------------------
-- 0. CSV PATH - EDIT THIS BEFORE RUNNING
------------------------------------------------------------
-- BULK INSERT reads from the SQL SERVER's filesystem, not the client's.
-- Point this at wherever ai4i2020.csv lives relative to your SQL Server
-- instance (a UNC path if the server is remote), then run the whole script
-- with sqlcmd, or paste this :setvar block at the top of your SSMS query
-- window before running the rest.

:setvar CsvPath "E:\teck\final\ai4i2020.csv"


------------------------------------------------------------
-- 1. CREATE DATABASE
------------------------------------------------------------

IF DB_ID('PredictiveMaintenanceDB') IS NULL
BEGIN
    CREATE DATABASE PredictiveMaintenanceDB;
END;
GO

USE PredictiveMaintenanceDB;
GO


------------------------------------------------------------
-- 2. CLEAN OLD TABLES
------------------------------------------------------------

DROP TABLE IF EXISTS dbo.PredictionHistory;
DROP TABLE IF EXISTS dbo.MachineReadings;
DROP TABLE IF EXISTS dbo.MachineType;
DROP TABLE IF EXISTS dbo.Stg_AI4I;
GO


------------------------------------------------------------
-- 3. STAGING TABLE
------------------------------------------------------------

CREATE TABLE dbo.Stg_AI4I
(
    UDI                    INT           NOT NULL,
    ProductID              VARCHAR(20)   NOT NULL,
    MachineType             CHAR(1)       NOT NULL,
    AirTemperatureK         DECIMAL(6,2)  NOT NULL,
    ProcessTemperatureK     DECIMAL(6,2)  NOT NULL,
    RotationalSpeedRPM      INT           NOT NULL,
    TorqueNm               DECIMAL(7,2)  NOT NULL,
    ToolWearMin             INT           NOT NULL,
    MachineFailure          BIT           NOT NULL,
    TWF                      BIT           NOT NULL,
    HDF                      BIT           NOT NULL,
    PWF                      BIT           NOT NULL,
    OSF                      BIT           NOT NULL,
    RNF                      BIT           NOT NULL
);
GO


------------------------------------------------------------
-- 4. LOAD CSV DATA
------------------------------------------------------------
-- CODEPAGE 65001 handles the UTF-8 BOM at the start of the source file
-- safely, in case a tool ever changes how FIRSTROW handles it.

BULK INSERT dbo.Stg_AI4I
FROM '$(CsvPath)'
WITH
(
    FIRSTROW = 2,
    FORMAT = 'CSV',
    FIELDQUOTE = '"',
    CODEPAGE = '65001'
);
GO


------------------------------------------------------------
-- 5. DATA VALIDATION
------------------------------------------------------------

-- 5.1 Number of rows

SELECT COUNT(*) AS TotalRows
FROM dbo.Stg_AI4I;


-- 5.2 Check duplicate UDI

SELECT
    UDI,
    COUNT(*) AS DuplicateCount
FROM dbo.Stg_AI4I
GROUP BY UDI
HAVING COUNT(*) > 1;


-- 5.3 Check machine types

SELECT
    MachineType,
    COUNT(*) AS TotalCount
FROM dbo.Stg_AI4I
GROUP BY MachineType;


-- 5.4 Check invalid values

SELECT *
FROM dbo.Stg_AI4I
WHERE AirTemperatureK <= 0
   OR ProcessTemperatureK <= 0
   OR RotationalSpeedRPM < 0
   OR TorqueNm < 0
   OR ToolWearMin < 0;
GO


------------------------------------------------------------
-- 6. MACHINE TYPE TABLE
------------------------------------------------------------

CREATE TABLE dbo.MachineType
(
    MachineTypeKey INT IDENTITY(1,1) PRIMARY KEY,
    MachineType    CHAR(1) NOT NULL UNIQUE,
    TypeEncoded    INT NOT NULL
);
GO


INSERT INTO dbo.MachineType
(
    MachineType,
    TypeEncoded
)
VALUES
('L', 0),
('M', 1),
('H', 2);
GO


------------------------------------------------------------
-- 7. MACHINE READINGS
-- Cleaned + feature-ready data
------------------------------------------------------------

CREATE TABLE dbo.MachineReadings
(
    ObservationID            INT IDENTITY(1,1) PRIMARY KEY,

    UDI                       INT NOT NULL UNIQUE,
    ProductID                 VARCHAR(20) NOT NULL UNIQUE,

    MachineTypeKey            INT NOT NULL,

    AirTemperatureK           DECIMAL(6,2) NOT NULL,
    ProcessTemperatureK       DECIMAL(6,2) NOT NULL,
    RotationalSpeedRPM        INT NOT NULL,
    TorqueNm                  DECIMAL(7,2) NOT NULL,
    ToolWearMin               INT NOT NULL,

    MachineFailure            BIT NOT NULL,
    TWF                       BIT NOT NULL,
    HDF                       BIT NOT NULL,
    PWF                       BIT NOT NULL,
    OSF                       BIT NOT NULL,
    RNF                       BIT NOT NULL,

    TempDiffK                 DECIMAL(8,3) NOT NULL,
    PowerW                    DECIMAL(12,3) NOT NULL,
    WearTorqueInteraction     DECIMAL(12,3) NOT NULL,

    LoadDate                  DATETIME2(0)
        DEFAULT SYSDATETIME(),

    CONSTRAINT FK_MachineReadings_MachineType
        FOREIGN KEY (MachineTypeKey)
        REFERENCES dbo.MachineType(MachineTypeKey)
);
GO


------------------------------------------------------------
-- 8. ETL
-- RAW -> CLEANED + FEATURE-READY
--
-- Note: TempDiffK / PowerW / WearTorqueInteraction use the exact same
-- formulas as src/features.py (Python) - keep the two in sync if either
-- changes.
------------------------------------------------------------

INSERT INTO dbo.MachineReadings
(
    UDI,
    ProductID,
    MachineTypeKey,

    AirTemperatureK,
    ProcessTemperatureK,
    RotationalSpeedRPM,
    TorqueNm,
    ToolWearMin,

    MachineFailure,
    TWF,
    HDF,
    PWF,
    OSF,
    RNF,

    TempDiffK,
    PowerW,
    WearTorqueInteraction
)

SELECT
    s.UDI,
    s.ProductID,
    mt.MachineTypeKey,

    s.AirTemperatureK,
    s.ProcessTemperatureK,
    s.RotationalSpeedRPM,
    s.TorqueNm,
    s.ToolWearMin,

    s.MachineFailure,
    s.TWF,
    s.HDF,
    s.PWF,
    s.OSF,
    s.RNF,

    -- Feature 1: temperature difference
    s.ProcessTemperatureK - s.AirTemperatureK,

    -- Feature 2: mechanical power = torque * angular speed (2*pi*rpm/60)
    (
        s.RotationalSpeedRPM
        * s.TorqueNm
        * 2
        * PI()
        / 60.0
    ),

    -- Feature 3: tool wear x torque interaction
    (
        s.ToolWearMin
        * s.TorqueNm
    )

FROM dbo.Stg_AI4I s

INNER JOIN dbo.MachineType mt
    ON s.MachineType = mt.MachineType;
GO


------------------------------------------------------------
-- 9. CHECK ETL RESULT
------------------------------------------------------------

SELECT COUNT(*) AS FeatureReadyRows
FROM dbo.MachineReadings;


SELECT TOP 10 *
FROM dbo.MachineReadings
ORDER BY ObservationID;
GO


------------------------------------------------------------
-- 10. PREDICTION HISTORY
------------------------------------------------------------
-- Populated by the Streamlit app's prediction log
-- (app/prediction_history.csv uses these exact column names).

CREATE TABLE dbo.PredictionHistory
(
    PredictionID       BIGINT IDENTITY(1,1) PRIMARY KEY,

    ObservationID      INT NULL,

    PredictionTime     DATETIME2(0)
        DEFAULT SYSDATETIME(),

    ModelName          VARCHAR(100) NOT NULL,

    ModelVersion       VARCHAR(50) NOT NULL,

    FailureProbability DECIMAL(8,6) NOT NULL,

    PredictedFailure   BIT NOT NULL,

    RiskTier           VARCHAR(20) NOT NULL,

    RecommendedAction  VARCHAR(100) NULL,

    ActualFailure      BIT NULL,

    CONSTRAINT FK_PredictionHistory_Observation
        FOREIGN KEY (ObservationID)
        REFERENCES dbo.MachineReadings(ObservationID),

    CONSTRAINT CK_PredictionHistory_Probability
        CHECK (FailureProbability BETWEEN 0 AND 1),

    CONSTRAINT CK_PredictionHistory_RiskTier
        CHECK (RiskTier IN ('Low', 'Medium', 'High'))
);
GO


-- Example prediction

INSERT INTO dbo.PredictionHistory
(
    ObservationID,
    ModelName,
    ModelVersion,
    FailureProbability,
    PredictedFailure,
    RiskTier,
    RecommendedAction,
    ActualFailure
)
VALUES
(
    1,
    'RandomForest',
    'v1.0',
    0.82,
    1,
    'High',
    'Inspect machine',
    NULL
);
GO


------------------------------------------------------------
-- 11. ANALYTICAL QUERIES
------------------------------------------------------------

------------------------------------------------------------
-- QUERY 1
-- Overall failure rate
------------------------------------------------------------

SELECT
    COUNT(*) AS TotalObservations,

    SUM(CAST(MachineFailure AS INT))
        AS FailureCount,

    CAST
    (
        100.0
        * SUM(CAST(MachineFailure AS INT))
        / COUNT(*)
        AS DECIMAL(6,2)
    )
        AS FailureRatePercent

FROM dbo.MachineReadings;
GO


------------------------------------------------------------
-- QUERY 2 - JOIN
-- Failure rate by machine type
------------------------------------------------------------

SELECT
    mt.MachineType,

    COUNT(*) AS TotalObservations,

    SUM(CAST(mr.MachineFailure AS INT))
        AS Failures,

    CAST
    (
        100.0
        * SUM(CAST(mr.MachineFailure AS INT))
        / COUNT(*)
        AS DECIMAL(6,2)
    )
        AS FailureRatePercent

FROM dbo.MachineReadings mr

INNER JOIN dbo.MachineType mt
    ON mr.MachineTypeKey = mt.MachineTypeKey

GROUP BY mt.MachineType

ORDER BY FailureRatePercent DESC;
GO


------------------------------------------------------------
-- QUERY 3 - CTE
-- Compare failed vs normal machines
------------------------------------------------------------

WITH FailureGroups AS
(
    SELECT
        MachineFailure,

        COUNT(*) AS Observations,

        AVG(AirTemperatureK)
            AS AvgAirTemperature,

        AVG(ProcessTemperatureK)
            AS AvgProcessTemperature,

        AVG(RotationalSpeedRPM)
            AS AvgRPM,

        AVG(TorqueNm)
            AS AvgTorque,

        AVG(ToolWearMin)
            AS AvgToolWear

    FROM dbo.MachineReadings

    GROUP BY MachineFailure
)

SELECT
    MachineFailure,
    Observations,
    AvgAirTemperature,
    AvgProcessTemperature,
    AvgRPM,
    AvgTorque,
    AvgToolWear

FROM FailureGroups

ORDER BY MachineFailure;
GO


------------------------------------------------------------
-- QUERY 4
-- Failure modes
------------------------------------------------------------

SELECT

    SUM(CAST(TWF AS INT))
        AS ToolWearFailures,

    SUM(CAST(HDF AS INT))
        AS HeatDissipationFailures,

    SUM(CAST(PWF AS INT))
        AS PowerFailures,

    SUM(CAST(OSF AS INT))
        AS OverstrainFailures,

    SUM(CAST(RNF AS INT))
        AS RandomFailures

FROM dbo.MachineReadings;
GO


------------------------------------------------------------
-- QUERY 5 - JOIN
-- Failure modes by machine type
------------------------------------------------------------

SELECT
    mt.MachineType,

    SUM(CAST(mr.TWF AS INT))
        AS TWF,

    SUM(CAST(mr.HDF AS INT))
        AS HDF,

    SUM(CAST(mr.PWF AS INT))
        AS PWF,

    SUM(CAST(mr.OSF AS INT))
        AS OSF,

    SUM(CAST(mr.RNF AS INT))
        AS RNF

FROM dbo.MachineReadings mr

INNER JOIN dbo.MachineType mt
    ON mr.MachineTypeKey = mt.MachineTypeKey

GROUP BY mt.MachineType

ORDER BY mt.MachineType;
GO


------------------------------------------------------------
-- QUERY 6 - WINDOW FUNCTION
-- Rank machines by tool wear
------------------------------------------------------------

SELECT

    mr.ObservationID,

    mr.ProductID,

    mt.MachineType,

    mr.ToolWearMin,

    RANK() OVER
    (
        PARTITION BY mt.MachineType
        ORDER BY mr.ToolWearMin DESC
    ) AS WearRank

FROM dbo.MachineReadings mr

INNER JOIN dbo.MachineType mt
    ON mr.MachineTypeKey = mt.MachineTypeKey

ORDER BY
    mt.MachineType,
    WearRank;
GO


------------------------------------------------------------
-- QUERY 7 - WINDOW FUNCTION
-- Top 5 highest-power observations
-- for each machine type
------------------------------------------------------------

WITH RankedPower AS
(
    SELECT

        mr.ObservationID,

        mr.ProductID,

        mt.MachineType,

        mr.PowerW,

        ROW_NUMBER() OVER
        (
            PARTITION BY mt.MachineType
            ORDER BY mr.PowerW DESC
        ) AS RowNum

    FROM dbo.MachineReadings mr

    INNER JOIN dbo.MachineType mt
        ON mr.MachineTypeKey = mt.MachineTypeKey
)

SELECT

    ObservationID,
    ProductID,
    MachineType,
    PowerW

FROM RankedPower

WHERE RowNum <= 5

ORDER BY
    MachineType,
    PowerW DESC;
GO


------------------------------------------------------------
-- QUERY 8
-- Average temperature difference
-- and power by machine type
------------------------------------------------------------

SELECT

    mt.MachineType,

    AVG(mr.TempDiffK)
        AS AvgTemperatureDifference,

    AVG(mr.PowerW)
        AS AvgPower,

    MIN(mr.PowerW)
        AS MinPower,

    MAX(mr.PowerW)
        AS MaxPower

FROM dbo.MachineReadings mr

INNER JOIN dbo.MachineType mt
    ON mr.MachineTypeKey = mt.MachineTypeKey

GROUP BY mt.MachineType

ORDER BY mt.MachineType;
GO


------------------------------------------------------------
-- QUERY 9 - CTE
-- Number of failure modes per observation
------------------------------------------------------------

WITH FailureModeCounts AS
(
    SELECT

        ObservationID,

        ProductID,

        MachineFailure,

        (
            CAST(TWF AS INT)
            + CAST(HDF AS INT)
            + CAST(PWF AS INT)
            + CAST(OSF AS INT)
            + CAST(RNF AS INT)
        ) AS FailureModeCount

    FROM dbo.MachineReadings
)

SELECT

    FailureModeCount,

    COUNT(*) AS ObservationCount

FROM FailureModeCounts

WHERE FailureModeCount > 0

GROUP BY FailureModeCount

ORDER BY FailureModeCount;
GO


------------------------------------------------------------
-- QUERY 10
-- Tool wear groups and failure rate
------------------------------------------------------------

SELECT

    CASE

        WHEN ToolWearMin < 100
            THEN '0-99'

        WHEN ToolWearMin < 200
            THEN '100-199'

        WHEN ToolWearMin < 300
            THEN '200-299'

        ELSE '300+'

    END AS ToolWearGroup,

    COUNT(*) AS Observations,

    SUM(CAST(MachineFailure AS INT))
        AS Failures,

    CAST
    (
        100.0
        * SUM(CAST(MachineFailure AS INT))
        / COUNT(*)
        AS DECIMAL(6,2)
    )
        AS FailureRatePercent

FROM dbo.MachineReadings

GROUP BY

    CASE

        WHEN ToolWearMin < 100
            THEN '0-99'

        WHEN ToolWearMin < 200
            THEN '100-199'

        WHEN ToolWearMin < 300
            THEN '200-299'

        ELSE '300+'

    END

ORDER BY
    MIN(ToolWearMin);
GO


------------------------------------------------------------
-- QUERY 11 - CTE + JOIN
-- Average power by machine type
------------------------------------------------------------

WITH TypePower AS
(
    SELECT

        mt.MachineType,

        AVG(mr.PowerW)
            AS AveragePower

    FROM dbo.MachineReadings mr

    INNER JOIN dbo.MachineType mt
        ON mr.MachineTypeKey = mt.MachineTypeKey

    GROUP BY mt.MachineType
)

SELECT

    MachineType,
    AveragePower

FROM TypePower

ORDER BY AveragePower DESC;
GO


------------------------------------------------------------
-- QUERY 12 - WINDOW FUNCTION
-- Running failure count
------------------------------------------------------------

SELECT

    ObservationID,

    ProductID,

    MachineFailure,

    SUM(CAST(MachineFailure AS INT))
    OVER
    (
        ORDER BY ObservationID

        ROWS BETWEEN
        UNBOUNDED PRECEDING
        AND CURRENT ROW
    )
        AS RunningFailureCount

FROM dbo.MachineReadings

ORDER BY ObservationID;
GO


------------------------------------------------------------
-- END OF SQL PART
------------------------------------------------------------
