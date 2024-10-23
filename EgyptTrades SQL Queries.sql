------------------------KPIs--------------------------------------------------------
--1. What are the products with the highest import value between 2005 and 2023?
--KPIs:
--Total annual import value for each product: To find out which products represent the highest import value.
--Percentage of each product in total imports: To determine the importance of each product compared to other products.

WITH essential_imports AS (
    SELECT t.year, SUM(t.import) AS  Total_Essential_Import_Value
    FROM   Trades t
    JOIN   Products p ON t.product_id = p.pid
    WHERE  p.product IN ('Crude oil', 'Natural gas', 'Wheat', 'Rice', 'Meat and preparations thereof') -- مثال على السلع الأساسية
    GROUP BY t.year),
total_imports AS (
    SELECT year, SUM(import) AS   Total_Import_Value
    FROM   Trades
    GROUP BY year )
SELECT  e.year, e.total_essential_import_value, 
        t.total_import_value,(e.total_essential_import_value / t.total_import_value * 100) AS Essential_Import_Percentage
FROM    essential_imports e
JOIN    total_imports t ON e.year = t.year
ORDER BY e.year;
---------------------
--2. What are the top geographic regions from which Egypt imports between 2005 and 2023?
--KPIs:
--Total value of imports per geographic region annually: To know the regions on which Egypt depends for imports.
--Percentage of each region of total imports: To determine the share of each geographic region.

SELECT   gr.region,gd.year, SUM(gd.import_value) AS Total_Import_Value,
         (SUM(gd.import_value) / (SELECT SUM(import_value) FROM geo_distro WHERE year = gd.year) * 100) AS Percentage_of_total_imports
FROM     geo_distro gd
JOIN     geo_regions gr ON gd.rid = gr.rid
GROUP BY gr.region, gd.year
ORDER BY gd.year, total_import_value DESC;
--------------------------------------------------------------------------------------
--3. How have imports of goods changed by manufacturing stage during the period from 2005 to 2023?
--KPIs:
--Total value of imports by manufacturing stage annually: To know the distribution of imports between raw materials, intermediate goods, and final goods.
--Percentage of each manufacturing stage of total imports: To determine the importance of each manufacturing stage in the import structure.
SELECT      ms.manu_stage,t.year,SUM(t.import) AS total_import_value,
            (SUM(t.import) / (SELECT SUM(import) FROM Trades WHERE year = t.year) * 100) AS percentage_of_total_imports
FROM        Trades t
     JOIN  Products p ON t.product_id = p.pid
     JOIN  manufacturing_stage ms ON p.stage_id = ms.stage_id
GROUP BY   ms.manu_stage, t.year
ORDER BY     t.year, total_import_value DESC;
-------------------------------------------------------------------------------------
--4. Which economic sectors import the largest value of goods from 2005 to 2023?
--KPIs:
--Total value of imports per sector per year: To identify the sectors most dependent on imports.
--Percentage of each sector in total imports: To know the share of each sector.
SELECT  s.sector, t.year,SUM(t.import) AS total_import_value,
       (SUM(t.import) / (SELECT SUM(import) FROM Trades WHERE year = t.year) * 100) AS percentage_of_total_imports
FROM   Trades t
    JOIN   Products p ON t.product_id = p.pid
    JOIN   Sectors s ON p.sid = s.sid
GROUP BY   s.sector, t.year
ORDER BY   t.year, total_import_value DESC;
-------------------------------------------------------------------------------------
--5. What is the impact of global economic crises (such as the 2008 crisis and the COVID-19 pandemic) on Egypt's imports?
--KPIs:
--Annual change in imports during crisis years: to know the extent to which imports were affected by crises.
--Percentage of decline or growth in imports during crisis years: to measure the relative impact of crises.
WITH yearly_imports AS (
    SELECT    year, SUM(import) AS total_import_value
    FROM      Trades
    GROUP BY  year)
SELECT   year,total_import_value,
         LAG(total_import_value) OVER (ORDER BY year) AS previous_year_import_value,
         (total_import_value - LAG(total_import_value) OVER (ORDER BY year)) AS difference_in_imports,
         ((total_import_value - LAG(total_import_value) OVER (ORDER BY year)) / LAG(total_import_value) OVER (ORDER BY year) * 100) AS percentage_change
FROM     yearly_imports
WHERE    year IN (2008, 2009, 2020, 2021)
ORDER BY year;
-------------------------------------------------------------------------------------
--6. Which products had the largest trade deficit during the period 2005-2023?
--KPIs:
--Surplus/deficit per product per year: To know which products contribute to the trade deficit.
--Percentage of trade deficit per product: To know which products contribute the most to the trade deficit.
SELECT  p.product,t.year, SUM(t.import - t.export) AS trade_deficit
FROM    Trades t
    JOIN  Products p ON t.product_id = p.pid
GROUP BY  p.product, t.year
HAVING   trade_deficit > 0
ORDER BY trade_deficit DESC;
------------------------------------------------------------------------------------
--7. What are the seasonal patterns of food imports over the years?
--KPIs:
--Average monthly and quarterly growth of imports by product: To identify seasonal patterns.
SELECT 
         t.year AS year, 
         SUM(t.import) AS total_import_value
FROM     Trades t
JOIN     Products p ON t.product_id = p.pid
WHERE     p.sid = (SELECT sid FROM Sectors WHERE sector = 'Foodstuff merchandise')
GROUP BY  year 
ORDER BY  year ;
--------------------------------------------------------------------------------------
--8. What are the differences in imports by year?
--KPIs:
--Yearly growth rate of imports: To see the annual increase or decrease in imports.
--Years of significant changes: Identify years that have seen significant changes in imports.
WITH yearly_imports AS (
    SELECT year,SUM(import) AS total_import_value
    FROM      Trades
    GROUP BY  year)
SELECT     year, total_import_value,
           LAG (total_import_value) OVER (ORDER BY year) AS previous_year_import_value,
               (total_import_value - LAG(total_import_value) OVER (ORDER BY year)) AS difference_in_imports,
              ((total_import_value - LAG(total_import_value) OVER (ORDER BY year)) / LAG(total_import_value) OVER (ORDER BY year) * 100) AS percentage_change
FROM     yearly_imports
ORDER BY year;
-----------------------------------------------------------------------------------------------
--9. Which sectors have seen their imports decline?
--KPIs:
--Year-on-year decline in imports for each sector: To find out which sectors have seen a decline in imports.
--Year-on-year decline rate: To determine the size of the decline for each sector.
WITH sector_imports AS (
    SELECT   s.sector,t.year,SUM(t.import) AS total_import_value
    FROM    Trades t
         JOIN Products p ON t.product_id = p.pid
         JOIN Sectors s ON p.sid = s.sid
    GROUP BY  s.sector, t.year),
sector_decline AS (
    SELECT 
        sector, year,  total_import_value,
        LAG(total_import_value) OVER (PARTITION BY sector ORDER BY year) AS previous_year_import_value,
        (total_import_value - LAG(total_import_value) OVER (PARTITION BY sector ORDER BY year)) AS difference_in_imports
     FROM  sector_imports)
SELECT    sector, year,total_import_value,difference_in_imports
FROM      sector_decline
WHERE     difference_in_imports < 0
ORDER BY    sector, year;
-------------------------------------------------------------------------------------------
--10. What is the ratio of commodity imports to total imports?
--KPIs:
--Commodity ratio to total imports: To find out the share of commodities in total imports.
--Year-on-year change in commodity ratio: To track changes in commodity imports over time.
WITH essential_imports AS (
    SELECT       t.year,  SUM(t.import) AS total_essential_import_value
    FROM         Trades t
    JOIN         Products p ON t.product_id = p.pid
    WHERE        p.product IN ('Crude oil', 'Natural gas', 'Wheat', 'Rice', 'Meat and preparations thereof') -- مثال على السلع الأساسية
    GROUP BY     t.year),
total_imports AS (
    SELECT     year, SUM(import) AS total_import_value
    FROM       Trades
    GROUP BY   year)
SELECT 
         e.year, e.total_essential_import_value, 
         t.total_import_value,
         (e.total_essential_import_value / t.total_import_value * 100) AS essential_import_percentage
FROM      essential_imports e
JOIN      total_imports t ON e.year = t.year
ORDER BY   e.year;
---------------------------------------------------------------------------------------
--11. Which sectors have seen growth in imports compared to other sectors?
--KPIs:
--Yearly growth rate for each sector: To identify sectors that have seen accelerated growth.
--Fastest growing sectors in imports: To identify sectors that are increasingly dependent on imports.
WITH sector_imports AS (
    SELECT  s.sector,t.year,SUM(t.import) AS total_import_value
    FROM    Trades t
    JOIN    Products p ON t.product_id = p.pid
    JOIN    Sectors s ON p.sid = s.sid
    GROUP BY s.sector, t.year),
sector_growth AS (
    SELECT    sector, year, total_import_value,
           LAG(total_import_value) OVER (PARTITION BY sector ORDER BY year) AS previous_year_import_value,
               (total_import_value - LAG(total_import_value) OVER (PARTITION BY sector ORDER BY year)) AS difference_in_imports,
               ((total_import_value - LAG(total_import_value) OVER (PARTITION BY sector ORDER BY year)) / LAG(total_import_value) OVER (PARTITION BY sector ORDER BY year) * 100) AS percentage_growth
    FROM      sector_imports)
SELECT   sector, year, total_import_value,difference_in_imports,percentage_growth
FROM     sector_growth
ORDER BY sector, year;





