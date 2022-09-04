# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 01:01:41 2020

@author: Qisen Ma

Barra 6 多因子模型收益率测算
"""
from rqdatac import init,is_st_stock,is_suspended,get_turnover_rate,\
    get_price,all_instruments,get_shares,get_factor,get_previous_trading_date,get_dividend,\
    get_trading_dates,get_instrument_industry,get_yield_curve
from numpy import std,log,multiply,array,isnan,exp,mean,identity,sqrt,zeros,insert,mat,dot
from numpy.linalg import inv
import pandas as pd
import statsmodels.api as sm
from datetime import date
from dateutil.relativedelta import relativedelta
import csv
import time

def winsorize(data):
    #其实在normalize后均值为0,方差标准差都为1,已经是定数了,这里为了一般化还是可以算一算,问题不大
    data = array(data)
    mean = data.mean()
    sigma = data.std()
    max = data.max()
    min = data.min()
    threeSigma = 3*sigma
    uScaler = (max-threeSigma-mean)/sigma*2
    dScaler = (mean-threeSigma-min)/sigma*2
    #如果数据都够好了就不处理咯
    if uScaler<=1 and dScaler<=1:
        return data
    index = len(data)
    for i in range(index):
        bias = data[i] - mean
        #太大了
        if bias > threeSigma:
            data[i] = (bias-threeSigma)/uScaler+threeSigma+mean
        else:
            #太小了
            if bias < -threeSigma:
                data[i] = (bias + threeSigma) / dScaler + mean - threeSigma
    return data
def normalize(data):
    '''
    std = data.std()[0]
    mean = data.mean()[0]
    '''
    std = array(data).std()
    mean = array(data).mean()
    if std != 0:
        data = (data - mean) / std
    else:
        data = data - mean
    return data

class barraFactor(object):
    def __init__(self, startTime, endTime):
        self.startTime = startTime
        self.endTime = endTime
        pass
    
    def getTradingDate(self):
        # 月末交易,构造一个交易时间list
        tradingDate = []
        startTimeStructure = self.startTime.split('.')
        endTimeStructure = self.endTime.split('.')
        startYear = int(startTimeStructure[0])
        startMonth = int(startTimeStructure[1])
        endYear = int(endTimeStructure[0])
        endMonth = int(endTimeStructure[1])
        dateTemp = [131, 227, 331, 430, 531, 630, 731, 831, 930, 1031, 1130, 1231]

        # 所有交易日
        #这里为了方便写死了开始日期,如果计算较早的历史数据,要修改哟
        self.marketTime = get_trading_dates(20010101, endYear*10000+dateTemp[endMonth])
        for year in range(startYear, endYear+1):
            for day in dateTemp:
                temp = date(year, day // 100, day % 100)
                #自然日月末那一天,如果是交易日,就考察它之前(不包含这一天)的因子,所以要取一个lag;如果这一天不是交易日,取两个lag就好了
                if temp in self.marketTime:
                    tradingDate.append(get_previous_trading_date(day + year * 10000, 1))
                else:
                    tradingDate.append(get_previous_trading_date(day + year * 10000, 2))
        #为了方便计算动量,这儿我们多取一个月
        if endMonth == 12:

            tradingDate = tradingDate[startMonth-1:]
        else:
            tradingDate = tradingDate[startMonth - 1:endMonth-12]
        return tradingDate
    
    #设置交易日信息
    def setTransferDay(self,transferDay):
        self.transferDay = transferDay

    #所有股票,去除退市的
    def getStockList(self):
        # 这儿会返回所有股票代码,包括st,科创板.
        stockList = all_instruments(type='CS')
        stockList = stockList.loc[stockList['status'] != 'Delisted']
        # 去除退市的
        self.stockListAll = stockList
        

    #上市时间
    def getListedDate(self):
        stockListAll = self.stockListAll

        listed_date = pd.DataFrame(data=list(stockListAll['listed_date']), index=stockListAll['order_book_id'])
        self.stockList = stockListAll.loc[stockListAll['listed_date'] < "20200101"]['order_book_id']
        return listed_date


if __name__=='__main__':

    print("Barra模型初始化。。。")
    init()
    
    FACTOR_YID = []
    dataDownload = barraFactor('2012.1','2013.6')
    tradingDate = dataDownload.getTradingDate()
    dataDownload.getStockList()
    stockListAll = dataDownload.stockListAll
    listed_date = dataDownload.getListedDate()
    priceAll = get_price(dataDownload.stockList, get_previous_trading_date(tradingDate[0], 520), tradingDate[-1], fields=['close'],expect_df=True)
    stockList = dataDownload.stockList
    T = len(tradingDate)
    f = open('barra6_result2.csv', 'w', newline='')
    dataCsv = csv.writer(f)
    dataCsv.writerow(["DATE","COUNTRY","FDC","GYSY","YYSW","DZ","YSJS","ZH","JXSB","SYMY","JZZS",
             "JYDQ","JTYS","QGZZ","QC","CM","HG","JSJ","TX","JZCL","XXFW","GFJG",
             "FYJR","DQSB","SPYL","NLMY","CJ","GT","FZFZ","YH",
             "LNCAP","MidCAP","BETA","HISTA","DAILYSTD","CULRANGE",
             "MONSHT","QUATSHT","ANNUALSHT","ANNUALTVR",
             "STR","SEASONAL","INDMOM","RSMOM","HISTA",
             "MLEV","BLEV","DTOA","VARS","VARE","VARCF","ABS","ACF",
             "ATO","GPROF","GPM","ROA","TAGR","ISSG","CEG",
             "BOTP","TEPR","CEP","EBITTOEV","LTRS","LTHA",
             "HEPSG","HSPSG","DIVYID"])

    #循环,每一个交易日计算一次
    for index in range(1,T-1):
        time_start=time.time()
        transferDay = tradingDate[index]
        transferDay_1 = tradingDate[index-1]
        transferDay_2 = tradingDate[index+1]
        dataDownload.setTransferDay(transferDay)
        shareAll = get_shares(stockList, transferDay, transferDay, 'circulation_a',expect_df=True)
        
        print("正在计算"+str(transferDay)+"的Barra因子")
        # st和停牌标志位
        stTag = is_st_stock(stockList, transferDay, transferDay)
        suspendTag = is_suspended(stockList, transferDay, transferDay)
        stockList_T = []
        
        ##波动率因子init
        
        daily_trading_dates = get_trading_dates(transferDay - relativedelta(years=6), transferDay)
        hs300_T = get_price('000300.XSHG', daily_trading_dates[-253], transferDay, fields=['close'],expect_df=True).loc['000300.XSHG'].close
        hs300_yid_T = array(hs300_T[1:])/array(hs300_T[:-1])-1
            
        beta_raw = []
        #这是动量里的alpha
        alpha_mom_raw = []
        hist_sigma_raw = []
        daily_std_raw = []
        cul_range_raw = []
        
        ##流动性因子
        month_share_tover_raw = []
        quarter_share_tover_raw = []
        annual_share_tover_raw = []
        annualized_share_tover_raw = []
        
        ##质量因子
        ##部分价值因子
        ##成长因子
        
        mlev_raw = []
        blev_raw = []
        dtoa_raw = []
        vars_raw = []
        vare_raw = []
        varcf_raw = []
        abs_raw = []
        acf_raw = []
        ato_raw = []
        gprof_raw = []
        gpm_raw = []
        roa_raw = []
        tagr_raw = []
        issg_raw = []
        ceg_raw = []
        
        botp_raw = []
        tepr_raw = []
        cep_raw = []
        ebittoev_raw = []
        
        hepsg_raw = []
        hspsg_raw = []
        
        ##长期价值因子
        ##动量因子
        
        ltrs_raw = []
        ltha_raw = []
        
        strev_raw = []
        sea_mom_raw = []
        strs_mom_raw = []
        
        hs300_T_l = get_price('000300.XSHG', daily_trading_dates[-1325], daily_trading_dates[-273], fields=['close'],expect_df=True).loc['000300.XSHG'].close
        hs300_yid_T_l = array(hs300_T_l[1:])/array(hs300_T_l[:-1])-1
        
        
        ##股息率因子
        divyid_raw = []
        trading_dates_year = get_trading_dates(transferDay - relativedelta(years=1), transferDay)
        trading_dates_month = get_trading_dates(transferDay - relativedelta(months=1), transferDay)

        
       #剔除当前交易日停牌及st股票
        for stock in stockList:
            if suspendTag[stock][0]:
                #print(stock + '停牌了')
                continue
            if stTag[stock][0]:
                #print(stock + '是st股票')
                continue
            # 这里用的自然日,(股价)调仓日前30天(自然日)/前365天(自然日)-1
            # 先切片,stock索引都一样
            # 可能有返回空的风险
            try:
                price = priceAll.loc[stock]
                price = price.loc[:transferDay]
                if len(price) == 0:
                    # 还没上市
                    continue
            except:
                #print(stock + '还没有上市')
                # 第一步loc报错的话,现在都还没上市
                continue
            
            try:
                circ_shares = get_shares(order_book_ids = stock, start_date = daily_trading_dates[-1260], end_date=transferDay)
                if len(circ_shares) < 1260:
                    #print(stock + '缺少流通股数据')
                    continue
            except:
                #print(stock + '缺少流通股数据')
                continue
            
            ##波动率因子筛选
            stock_T = get_price(stock, daily_trading_dates[-253], transferDay, fields=['close'],expect_df=True).loc[stock].close
            try:      
                stock_yid_T = array(stock_T[1:])/array(stock_T[:-1])-1
                stock_yid_df = pd.DataFrame()
                stock_yid_df['stock_yid'] = stock_yid_T[::-1]
                stock_yid_half_63 = array(stock_yid_df.ewm(span = 63,ignore_na=False, adjust= True).mean().stock_yid)[::-1]
                est_stk_vol = sm.OLS(stock_yid_half_63, sm.add_constant(hs300_yid_T)).fit()
            except:
                #print(stock + '缺少波动率数据')
                continue           
            ##流动性因子筛选
            try:
                stock_turnover = array(get_turnover_rate(stock, daily_trading_dates[-252], transferDay, 'today', expect_df=True).loc[stock].today)
            except:
                #print(stock + '缺少换手率数据')
                continue
            
            if(len(stock_turnover)<252 or isnan(stock_turnover).any()):
                #print(stock + '缺少换手率数据')
                continue
            if sum(stock_turnover[-21:]) == 0:
                #print(stock + '缺少换手率数据')
                continue
            turn_flag = False
            for i in range(11):
                if sum(stock_turnover[-21*(i+2):-21*(i+1)])==0:
                    turn_flag = True
            if turn_flag:
                #print(stock + '缺少换手率数据')
                continue
            
            ##质量因子筛选
            try:
                factor_T = get_factor(stock, ['equity_preferred_stock_lyr_0','long_term_loans_lyr_0',
                                              'bond_payable_lyr_0','other_non_current_liabilities_lyr_0',
                                              'total_equity_lyr_0','total_liabilities_lyr_0',
                                              'cash_equivalent_lyr_0','long_term_loans_lyr_1',
                                              'cash_equivalent_ttm_0','minority_interest_ttm_0',
                                              'bond_payable_lyr_1','other_non_current_liabilities_lyr_1',
                                              'total_liabilities_lyr_1','cash_equivalent_lyr_1','short_term_loans_lyr_0',
                                              'short_term_loans_lyr_1','short_term_debt_lyr_0','short_term_debt_lyr_1',
                                              'long_term_liabilities_due_one_year_lyr_0','long_term_liabilities_due_one_year_lyr_1',
                                              'fixed_asset_depreciation_lyr_0','total_liabilities_ttm_0',
                                              'deferred_expense_amortization_lyr_0','intangible_asset_amortization_lyr_0',
                                              'cash_flow_from_operating_activities_lyr_0','cash_flow_from_investing_activities_lyr_0',
                                              'cost_of_goods_sold','operating_revenue_lyr_0','net_profit_ttm_0','total_assets_ttm_0',
                                              'cash_flow_from_operating_activities_ttm_0','cash_flow_from_investing_activities_ttm_0',
                                              'cash_flow_from_financing_activities_ttm_0','ebit_ttm','total_assets_mrq_0'],transferDay, transferDay)
                sales_T = get_factor(stock, ['operating_revenue_lyr_0','operating_revenue_lyr_1',
                                              'operating_revenue_lyr_2','operating_revenue_lyr_3',
                                              'operating_revenue_lyr_4'], transferDay, transferDay)
                earnings_T = get_factor(stock, ['net_profit_lyr_0','net_profit_lyr_1',
                                              'net_profit_lyr_2','net_profit_lyr_3',
                                              'net_profit_lyr_4'], transferDay, transferDay)
                cashflows_T = get_factor(stock, ['cash_equivalent_increase_lyr_0','cash_equivalent_increase_lyr_1',
                                              'cash_equivalent_increase_lyr_2','cash_equivalent_increase_lyr_3',
                                              'cash_equivalent_increase_lyr_4'], transferDay, transferDay)
                asset_T = get_factor(stock, ['total_assets_lyr_0','total_assets_lyr_1',
                                              'total_assets_lyr_2','total_assets_lyr_3',
                                              'total_assets_lyr_4'], transferDay, transferDay)
                capexp_T = get_factor(stock, ['cash_paid_for_asset_lyr_0','cash_paid_for_asset_lyr_1',
                                              'cash_paid_for_asset_lyr_2','cash_paid_for_asset_lyr_3',
                                              'cash_paid_for_asset_lyr_4','cash_received_from_disposal_of_asset_lyr_0',
                                              'cash_received_from_disposal_of_asset_lyr_1','cash_received_from_disposal_of_asset_lyr_2',
                                              'cash_received_from_disposal_of_asset_lyr_3','cash_received_from_disposal_of_asset_lyr_4'], transferDay, transferDay)
                circ_T = get_shares(order_book_ids = stock, start_date = daily_trading_dates[-1260], end_date=transferDay, fields='circulation_a')
            except:
                #print(stock+'缺少财务数据')
                continue
                
            label_T = True
            for factors in sales_T.keys():
                if sales_T[factors][0] != sales_T[factors][0]:
                    label_T = False
            for factors in earnings_T.keys():
                if earnings_T[factors][0] != earnings_T[factors][0]:
                    label_T = False
            for factors in cashflows_T.keys():
                if cashflows_T[factors][0] != cashflows_T[factors][0]:
                    label_T = False
            for factors in asset_T.keys():
                if asset_T[factors][0] != asset_T[factors][0]:
                    label_T = False
            for factors in capexp_T.keys():
                if capexp_T[factors][0] != capexp_T[factors][0]:
                    label_T = False
            
            dno_list = ['total_assets_ttm_0','total_assets_mrq_0','operating_revenue_lyr_0']
            for factors in dno_list:
                if factor_T[factors][0] != factor_T[factors][0]:
                    label_T = False
            
            if not label_T:
                #print(stock+'缺少财务数据')
                continue
            
            ##长期价值因子筛选
            stock_T_l = get_price(stock, daily_trading_dates[-1400], transferDay, fields=['close'],expect_df=True).loc[stock].close
            if len(stock_T_l) < 1400:
                #print(stock + "缺少长期价值数据")
                continue
            
            ##股息率因子筛选
            try:
                divd_df = get_dividend(stock, start_date=trading_dates_year[0], end_date=transferDay, market='cn')
                dividends_year = divd_df['dividend_cash_before_tax'][0] / divd_df['round_lot'][0]
                price_month = get_price(stock, trading_dates_month[0], trading_dates_month[0], fields=['close'],expect_df=True).loc[stock].close[0]
                div_factor = dividends_year / price_month
            except:
                div_factor = 0
                
            divyid_raw.append(div_factor)
            ##波动率因子计算
            beta_raw.append(est_stk_vol.params[1])
            alpha_mom_raw.append(est_stk_vol.params[0])
            res_T = stock_yid_half_63 - est_stk_vol.params[0] - multiply(est_stk_vol.params[1],hs300_yid_T)
            hist_sigma_raw.append(res_T.std())
            stock_yid_half_42 = array(stock_yid_df.ewm(span = 42,ignore_na=False, adjust= True).mean().stock_yid)[::-1]
            daily_std_raw.append(stock_yid_half_42.std())

            cul_stock_yield = []
            cul_range_stock = []
            for i in range(12):
                cul_stock_yield.append(log(stock_T[(i+1)*21]/stock_T[i*21+1]))
                cul_range_stock.append(sum(cul_stock_yield))
            cul_range_raw.append(max(cul_range_stock)-min(cul_range_stock))

            ##流动性因子计算
            stom_stock_list = [log(sum(stock_turnover[-21:]))]
            for i in range(11):
                stom_stock_i = log(sum(stock_turnover[-21*(i+2):-21*(i+1)]))
                stom_stock_list.insert(0,stom_stock_i)
                
            month_share_tover_raw.append(stom_stock_list[-1])
            quarter_share_tover_raw.append(log(sum(exp(stom_stock_list[-3:])) / 3))
            annual_share_tover_raw.append(log(sum(exp(stom_stock_list)) / 12))
            
            stock_to_df = pd.DataFrame()
            stock_to_df['stock_to'] = stock_turnover[::-1]
            stock_to_half_63 = array(stock_to_df.ewm(span = 63,ignore_na=False, adjust= True).mean().stock_to)[::-1]

            annualized_share_tover_raw.append(sum(stock_to_half_63))
            
            ##质量因子计算
            for factors in factor_T.keys():
                if factor_T[factors][0] != factor_T[factors][0]:
                    factor_T[factors][0] = 0
                    
            ME_T = shareAll.loc[stock,transferDay][0]*priceAll.loc[stock,transferDay][0]
            PE_T = factor_T.equity_preferred_stock_lyr_0[0]
            LD_T = factor_T.long_term_loans_lyr_0[0] + factor_T.bond_payable_lyr_0[0] + factor_T.other_non_current_liabilities_lyr_0[0]
            BE_T = factor_T.total_equity_lyr_0[0] - PE_T
            TL_T = factor_T.total_liabilities_lyr_0[0]
            TA_T = asset_T.total_assets_lyr_0[0]
            TA_mrq = factor_T.total_assets_mrq_0[0]
            Sales = factor_T.operating_revenue_lyr_0[0]
            COGS_T = factor_T.cost_of_goods_sold[0]
           
            MLEV_T = (ME_T+PE_T+LD_T) / ME_T
            BLEV_T = (BE_T+PE_T+LD_T) / ME_T
            DTOA_T = TL_T / TA_T
            
            sales_data = [sales_T.operating_revenue_lyr_4[0],sales_T.operating_revenue_lyr_3[0],
                          sales_T.operating_revenue_lyr_2[0],sales_T.operating_revenue_lyr_1[0],
                          sales_T.operating_revenue_lyr_0[0]]
            Var_in_Sales = std(sales_data) / mean(sales_data)
            
            earnings_data = [earnings_T.net_profit_lyr_4[0],earnings_T.net_profit_lyr_3[0],
                          earnings_T.net_profit_lyr_2[0],earnings_T.net_profit_lyr_1[0],
                          earnings_T.net_profit_lyr_0[0]]
            Var_in_Earnings = std(earnings_data) / mean(earnings_data)
            
            cashflows_data = [cashflows_T.cash_equivalent_increase_lyr_4[0],cashflows_T.cash_equivalent_increase_lyr_3[0],
                          cashflows_T.cash_equivalent_increase_lyr_2[0],cashflows_T.cash_equivalent_increase_lyr_1[0],
                          cashflows_T.cash_equivalent_increase_lyr_0[0]]
            Var_in_Cashflows = std(cashflows_data) / mean(cashflows_data)
            
            TD_T = LD_T + factor_T.short_term_loans_lyr_0[0] + factor_T.short_term_debt_lyr_0[0] + factor_T.long_term_liabilities_due_one_year_lyr_0[0]
            TD_1 = factor_T.long_term_loans_lyr_1[0] + factor_T.bond_payable_lyr_1[0] + factor_T.other_non_current_liabilities_lyr_1[0] + factor_T.short_term_loans_lyr_1[0] + factor_T.short_term_debt_lyr_1[0] + factor_T.long_term_liabilities_due_one_year_lyr_1[0]
            NOA_T = TA_T - factor_T.cash_equivalent_lyr_0[0] - TL_T + TD_T
            NOA_T_1 = asset_T.total_assets_lyr_1[0] - factor_T.cash_equivalent_lyr_1[0] - factor_T.total_liabilities_lyr_1[0] + TD_1
            DA_T = factor_T.fixed_asset_depreciation_lyr_0[0] + factor_T.deferred_expense_amortization_lyr_0[0] + factor_T.intangible_asset_amortization_lyr_0[0]
            ABS_T = -(NOA_T - NOA_T_1 - DA_T) / TA_T
            ACF_T = -(earnings_data[-1] - factor_T.cash_flow_from_operating_activities_lyr_0[0] - factor_T.cash_flow_from_investing_activities_lyr_0[0] + DA_T) / TA_T
            ATO_T = Sales / TA_mrq
            GP_T = (Sales - COGS_T) / TA_T
            GMP_T = (Sales - COGS_T) / Sales
            ROA = factor_T.net_profit_ttm_0[0] / factor_T.total_assets_ttm_0[0]
            
            TA_data = [asset_T.total_assets_lyr_4[0],asset_T.total_assets_lyr_3[0],
                          asset_T.total_assets_lyr_2[0],asset_T.total_assets_lyr_1[0],
                          asset_T.total_assets_lyr_0[0]]
            
            capexp_data = [capexp_T.cash_paid_for_asset_lyr_4[0]-capexp_T.cash_received_from_disposal_of_asset_lyr_4[0],
                           capexp_T.cash_paid_for_asset_lyr_3[0]-capexp_T.cash_received_from_disposal_of_asset_lyr_3[0],
                           capexp_T.cash_paid_for_asset_lyr_2[0]-capexp_T.cash_received_from_disposal_of_asset_lyr_2[0],
                           capexp_T.cash_paid_for_asset_lyr_1[0]-capexp_T.cash_received_from_disposal_of_asset_lyr_1[0],
                           capexp_T.cash_paid_for_asset_lyr_0[0]-capexp_T.cash_received_from_disposal_of_asset_lyr_0[0]]
            
            year_array = [1,2,3,4,5]
            
            est_TA = sm.OLS(TA_data, sm.add_constant(year_array)).fit()
            est_capexp = sm.OLS(capexp_data, sm.add_constant(year_array)).fit()
            
            TAGR = -est_TA.params[1] / mean(TA_data)
            
            circ_annual = []
            for i in range(5):
                circ_annual.append(mean(circ_T[i:252*i-1]))
            est_circ = sm.OLS(circ_annual, sm.add_constant(year_array)).fit()
            ISSG = -est_circ.params[1] / mean(circ_annual)
            
            CapexpG = -est_capexp.params[1] / mean(capexp_data)
            
            mlev_raw.append(MLEV_T)
            blev_raw.append(BLEV_T)
            dtoa_raw.append(DTOA_T)
            vars_raw.append(Var_in_Sales)
            vare_raw.append(Var_in_Earnings)
            varcf_raw.append(Var_in_Cashflows)
            abs_raw.append(ABS_T)
            acf_raw.append(ACF_T)
            ato_raw.append(ATO_T)
            gprof_raw.append(GP_T)
            gpm_raw.append(GMP_T)
            roa_raw.append(ROA)
            tagr_raw.append(TAGR)
            issg_raw.append(ISSG)
            ceg_raw.append(CapexpG)
            
            #价值因子
            BOTP_T = BE_T / ME_T
            TEPR_T = ME_T / factor_T.net_profit_ttm_0[0]
            CEP_T = (factor_T.cash_flow_from_operating_activities_ttm_0[0] + factor_T.cash_flow_from_investing_activities_ttm_0[0] + factor_T.cash_flow_from_financing_activities_ttm_0[0]) / ME_T
            EV_T = ME_T + factor_T.total_liabilities_ttm_0[0] + PE_T + factor_T.minority_interest_ttm_0[0] - factor_T.cash_equivalent_ttm_0[0]
            EBITtoEV = factor_T.ebit_ttm[0] / EV_T
            
            botp_raw.append(BOTP_T)
            tepr_raw.append(TEPR_T)
            cep_raw.append(CEP_T)
            ebittoev_raw.append(EBITtoEV)
            
            #成长因子
            EPS_T = array(sales_data) / array(circ_annual)
            SPS_T = array(earnings_data) / array(circ_annual)
            est_eps = sm.OLS(EPS_T, sm.add_constant(year_array)).fit()
            est_sps = sm.OLS(SPS_T, sm.add_constant(year_array)).fit()
            EPSG_T = est_eps.params[1] / mean(EPS_T)
            SPSG_T = est_sps.params[1] / mean(SPS_T)
            
            hepsg_raw.append(EPSG_T)
            hspsg_raw.append(SPSG_T)
            
            ##长期价值因子计算
            ##动量因子计算
            stock_yid_T_l5 = array(stock_T_l[1:])/array(stock_T_l[:-1])-1
            stock_yid_T_l = stock_yid_T_l5[:-273]
            ltrs = []
            ltha = []
            for i in range(11):
                stk_yid_lt = array(stock_yid_T_l[-1040-i-1:-i-1])
                ln_stk_yid = log(stk_yid_lt+1)
                ln_stock_yid_df = pd.DataFrame()
                ln_stock_yid_df['stock_yid'] = ln_stk_yid[::-1]
                ln_stock_yid_half_260 = array(ln_stock_yid_df.ewm(span = 260,ignore_na=False, adjust= True).mean().stock_yid)[::-1]
                ltrs.append(sum(ln_stock_yid_half_260))
                
                stock_yid_df = pd.DataFrame()
                stock_yid_df['stock_yid'] = stk_yid_lt[::-1]
                stock_yid_half_260 = array(stock_yid_df.ewm(span = 260,ignore_na=False, adjust= True).mean().stock_yid)[::-1]

                est_stk_l = sm.OLS(stock_yid_half_260, sm.add_constant(hs300_yid_T_l[-1040-i-1:-i-1])).fit()
                ltha.append(est_stk_l.params[0])
            ltrs_raw.append(-mean(ltrs))
            ltha_raw.append(-mean(ltha))
            
            ln_stk_yidp = log(array(stock_yid_T_l[-21:])+1)
            stock_yidp_df = pd.DataFrame()
            stock_yidp_df['stock_yidp'] = ln_stk_yidp[::-1]
            stock_yidp_half_5 = array(stock_yidp_df.ewm(span = 5,ignore_na=False, adjust= True).mean().stock_yidp)[::-1]
            strev = mean(stock_yidp_half_5)
            strev_raw.append(strev)
            
            month_returns = [stock_yid_T_l5[-1]]
            for i in range(1,60):
                month_returns.append(stock_yid_T_l5[-21*i])
            SEASON = mean(month_returns)
            
            strs = []
            for i in range(11):
                ln_stk_yid = log(array(stock_yid_T_l[-252-i-1:-i-1])+1)
                stock_yid_df = pd.DataFrame()
                stock_yid_df['stock_yid'] = ln_stk_yid[::-1]
                stock_yid_half_126 = array(stock_yid_df.ewm(span = 126,ignore_na=False, adjust= True).mean().stock_yid)[::-1]
                strs.append(sum(stock_yid_half_126))
                
            sea_mom_raw.append(SEASON)  
            strs_mom_raw.append(mean(strs))
            
            stockList_T.append(stock)
            
        stk_list_len = len(stockList_T)
        print(str(transferDay)+"的stock_size为"+str(stk_list_len))
        print("正在计算"+str(transferDay)+"的行业因子")
        ##市值因子
        ##行业因子及市值
        ##行业动量
        
        ind_size = {}
        ind_yid_tmp = {}
        size_T = []
        
        rs_mom = []
        rs_ind_mom = {}
        sqrt_size = []
        
        for stock in stockList_T:
            stk_share = shareAll.loc[stock,transferDay][0]
            stk_price = priceAll.loc[stock,transferDay][0]
            stk_price_1 = priceAll.loc[stock,transferDay_1][0]
            stk_yid_1 = stk_price / stk_price_1 - 1 
            stk_ind = get_instrument_industry(stock, source='sws', level=1, date=None, market='cn').first_industry_name[0]
            stk_size = stk_share*stk_price
            try:
                ind_size[stk_ind] += stk_size
            except:
                ind_size[stk_ind] = 0  
            try:
                ind_yid_tmp[stk_ind] += stk_size * stk_yid_1
            except:
                ind_yid_tmp[stk_ind] = 0
            
            size_T.append(log(stk_size))
            
            stock_T = get_price(stock, daily_trading_dates[-127], transferDay, fields=['close'],expect_df=True).loc[stock].close
            stock_yid_T_ln = log(array(stock_T[1:])/array(stock_T[:-1])+1)
            stock_yid_df = pd.DataFrame()
            stock_yid_df['stock_yid_ln'] = stock_yid_T_ln[::-1]
            stk_rs = sum(array(stock_yid_df.ewm(span = 21,ignore_na=False, adjust= True).mean().stock_yid_ln)[::-1])
            rs_mom.append(stk_rs)
            try:
                rs_ind_mom[stk_ind] += stk_rs*sqrt(stk_size)
            except:
                rs_ind_mom[stk_ind] = 0
            sqrt_size.append(sqrt(stk_size))
        
        ind_list = list(ind_size.keys())
        ind_factor_blank = {}
        for inds in ind_list:
            ind_factor_blank[inds] = 0
            
        ind_mom_raw =[]
        ind_factor_raw = []
        
        for i in range(stk_list_len):
            
            stock = stockList_T[i]
            stk_ind = get_instrument_industry(stock, source='sws', level=1, date=None, market='cn').first_industry_name[0]
            stk_ind_factor = dict.copy(ind_factor_blank)
            stk_ind_factor[stk_ind] = 1
            ind_factor_raw.append(stk_ind_factor)
            
            INDMOM = rs_ind_mom[stk_ind] - sqrt_size[i]*rs_mom[i]
            ind_mom_raw.append(INDMOM)
            
        size_T_3 = multiply(multiply(size_T,size_T),size_T)
        #市值中性化：size_T立方对size_T回归，取残差
        est = sm.OLS(size_T_3, sm.add_constant(size_T)).fit()
        MidCAP_raw = size_T_3 - est.params[0] - multiply(est.params[1],size_T)
        
        LNCAP = winsorize(array(size_T))
        MidCAP = winsorize(normalize(MidCAP_raw))
        
        BETA = winsorize(normalize(beta_raw))
        HISTS = winsorize(normalize(hist_sigma_raw))
        DAILYSTD= winsorize(normalize(daily_std_raw))
        CULRANGE= winsorize(normalize(cul_range_raw))
        
        MONSHT= winsorize(normalize(month_share_tover_raw))
        QUATSHT= winsorize(normalize(quarter_share_tover_raw))
        ANNUALSHT= winsorize(normalize(annual_share_tover_raw))
        ANNUALTVR= winsorize(normalize(annualized_share_tover_raw))
        
        STR= winsorize(normalize(strev_raw))
        SEASONAL= winsorize(normalize(sea_mom_raw))
        INDMOM= winsorize(normalize(ind_mom_raw))
        RSMOM= winsorize(normalize(strs_mom_raw))
        HISTA= winsorize(normalize(alpha_mom_raw))
        
        MLEV= winsorize(normalize(mlev_raw))
        BLEV= winsorize(normalize(blev_raw))
        DTOA= winsorize(normalize(dtoa_raw))
        VARS= winsorize(normalize(vars_raw))
        VARE= winsorize(normalize(vare_raw))
        VARCF= winsorize(normalize(varcf_raw))
        ABS= winsorize(normalize(abs_raw))
        ACF= winsorize(normalize(acf_raw))
        ATO= winsorize(normalize(ato_raw))
        GPROF= winsorize(normalize(gprof_raw))
        GPM= winsorize(normalize(gpm_raw))
        ROA= winsorize(normalize(roa_raw))
        TAGR= winsorize(normalize(tagr_raw))
        ISSG= winsorize(normalize(issg_raw))
        CEG= winsorize(normalize(ceg_raw))
        
        BOTP= winsorize(normalize(botp_raw))
        TEPR= winsorize(normalize(tepr_raw))
        CEP= winsorize(normalize(cep_raw))
        EBITTOEV= winsorize(normalize(ebittoev_raw))
        LTRS = winsorize(normalize(ltrs_raw))
        LTHA = winsorize(normalize(ltha_raw))
        
        HEPSG = winsorize(normalize(hepsg_raw))
        HSPSG = winsorize(normalize(hspsg_raw))

        DIVYID = winsorize(normalize(divyid_raw))

        ##计算矩阵
        P = len(ind_list)
        Q = 39
        K = 1 + P + Q
        N = len(stockList_T)
        R = identity(K-1)
        ind_base_size = ind_size[ind_list[-1]]
        line_P = zeros(K-1)
        for i in range(P-1):
            line_P[i+1] = -ind_size[ind_list[i]] / ind_base_size
        R = insert(R,[P-1],line_P,axis=0)
        R = mat(array(R))
        sqrt_size_sum = sum(sqrt_size)
        V = identity(N)
        for i in range(N):
            V[i][i] = sqrt_size[i] / sqrt_size_sum
        V = mat(V)
        #计算X矩阵
        COUNTRY = array([1] * N)
        FDC = []
        GYSY = []
        YYSW = []
        DZ = []
        YSJS = []
        ZH = []
        JXSB = []
        SYMY = []
        JZZS = []
        JYDQ = []
        JTYS = []
        QGZZ = []
        QC = []
        CM = []
        HG = []
        JSJ = []
        TX = []
        JZCL = []
        XXFW = []
        GFJG = []
        FYJR = []
        DQSB = []
        SPYL = []
        NLMY = []
        CJ = []
        GT = []
        FZFZ = []
        YH = []
        for ind_factor in ind_factor_raw:
            FDC.append(ind_factor['房地产'])
            GYSY.append(ind_factor['公用事业'])
            YYSW.append(ind_factor['医药生物'])
            DZ.append(ind_factor['电子'])
            YSJS.append(ind_factor['有色金属'])
            ZH.append(ind_factor['综合'])
            JXSB.append(ind_factor['机械设备'])
            SYMY.append(ind_factor['商业贸易'])
            JZZS.append(ind_factor['建筑装饰'])
            JYDQ.append(ind_factor['家用电器'])
            JTYS.append(ind_factor['交通运输'])
            QGZZ.append(ind_factor['轻工制造'])
            QC.append(ind_factor['汽车'])
            CM.append(ind_factor['传媒'])
            HG.append(ind_factor['化工'])
            JSJ.append(ind_factor['计算机'])
            TX.append(ind_factor['通信'])
            JZCL.append(ind_factor['建筑材料'])
            XXFW.append(ind_factor['休闲服务'])
            GFJG.append(ind_factor['国防军工'])
            FYJR.append(ind_factor['非银金融'])
            DQSB.append(ind_factor['电气设备'])
            SPYL.append(ind_factor['食品饮料'])
            NLMY.append(ind_factor['农林牧渔'])
            CJ.append(ind_factor['采掘'])
            GT.append(ind_factor['钢铁'])
            FZFZ.append(ind_factor['纺织服装'])
            YH.append(ind_factor['银行'])
        FDC = array(FDC)
        GYSY = array(GYSY)
        YYSW = array(YYSW)
        DZ = array(DZ)
        YSJS = array(YSJS)
        ZH = array(ZH)
        JXSB = array(JXSB)
        SYMY = array(SYMY)
        JZZS = array(JZZS)
        JYDQ = array(JYDQ)
        JTYS = array(JTYS)
        QGZZ = array(QGZZ)
        QC = array(QC)
        CM = array(CM)
        HG = array(HG)
        JSJ = array(JSJ)
        TX = array(TX)
        JZCL = array(JZCL)
        XXFW = array(XXFW)
        GFJG = array(GFJG)
        FYJR = array(FYJR)
        DQSB = array(DQSB)
        SPYL = array(SPYL)
        NLMY = array(NLMY)
        CJ = array(CJ)
        GT = array(GT)
        FZFZ = array(FZFZ)
        YH = array(YH)
        
        X = array([COUNTRY,FDC,GYSY,YYSW,DZ,YSJS,ZH,JXSB,SYMY,JZZS,
             JYDQ,JTYS,QGZZ,QC,CM,HG,JSJ,TX,JZCL,XXFW,GFJG,
             FYJR,DQSB,SPYL,NLMY,CJ,GT,FZFZ,YH,
             LNCAP,MidCAP,BETA,HISTS,DAILYSTD,CULRANGE,
             MONSHT,QUATSHT,ANNUALSHT,ANNUALTVR,
             STR,SEASONAL,INDMOM,RSMOM,HISTA,
             MLEV,BLEV,DTOA,VARS,VARE,VARCF,ABS,ACF,
             ATO,GPROF,GPM,ROA,TAGR,ISSG,CEG,
             BOTP,TEPR,CEP,EBITTOEV,LTRS,LTHA,
             HEPSG,HSPSG,DIVYID]).T
        
        OMEGA = dot(dot(dot(dot(R,inv(dot(dot(dot(dot(R.T,X.T),V),X),R))),R.T),X.T),V) 
        
        risk_free = get_yield_curve(transferDay_1).loc[transferDay_1]['1M']
        
        STK_YID_ALL = []
        
        for stock in stockList_T:
            price_stk = priceAll.loc[stock]
            price = price_stk.loc[transferDay_2][0] 
            price_1 = price_stk.loc[transferDay][0]
            stk_yid_tmp = price / price_1 - 1 - risk_free
            STK_YID_ALL.append(stk_yid_tmp)
            
        FACTOR_YID_T = OMEGA * mat(STK_YID_ALL).T
        FACTOR_YID.append(FACTOR_YID_T)
        
        mats_tmp = list(array(FACTOR_YID_T.T)[0])
        mats_tmp.insert(0,transferDay)
        dataCsv.writerow(mats_tmp)
        time_end=time.time()
        print(str(transferDay)+"的Barra因子计算完成，耗时"+str(float(time_end-time_start)/60)+"min，进入下一个月\n")
    
    f.close()