#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 15/12/2022 
# version ='1.1'
# ---------------------------------------------------------------------------
""" The toolkit to read and write a .xls file"""  

import xlwt
import numpy
class data_writer(object):

    # --------------------------------------------------------------------------------
    #initailization of the class
    # input variables:
    # book_name (string): the name of the data file
    # --------------------------------------------------------------------------------
    def __init__(self,book_name):
        self.wb = xlwt.Workbook()
        self.book_name = book_name
        self.sheet_name_list = []
        self.sheet_list = []
        pass
    
    # --------------------------------------------------------------------------------
    # function: find_sheet
    # find the sheet 
    # input variables:
    # sheet_name (string): the name of the sheet
    # return:
    # the sheet with the name given by sheet_name
    #-------------------------------------------------------------------------------- 
    def find_sheet(self,sheet_name):
        if sheet_name not in self.sheet_name_list:
            self.sheet_list.append(self.wb.add_sheet(sheet_name)) 
            self.sheet_name_list.append(sheet_name)
        
        return self.sheet_list[self.sheet_name_list.index(sheet_name)]

    # --------------------------------------------------------------------------------
    # function: single_data_w
    # write a single data
    # input variables:
    # data : the value of the data
    # loc_x , loc_y (int): the cell to write
    # sheet_name (string): the sheet to write the data
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def single_data_w(self,data,loc_x,loc_y,sheet_name):
        work_sheet = self.find_sheet(sheet_name)
        work_sheet.write(loc_x,loc_y,data)
        
        pass
    
    # --------------------------------------------------------------------------------
    # function: list_column_w
    # write a column of data
    # input variables:
    # data : the list of the values of the data
    # loc_y (int): the column to write
    # sheet_name (string): the sheet to write the data
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def list_column_w(self,data,loc_y,sheet_name):
        assert type(data) is list or numpy.ndarray
        work_sheet = self.find_sheet(sheet_name)
        for loc_x in range(0,len(data)):
            work_sheet.write(loc_x+1,loc_y,data[loc_x])
        pass


    # --------------------------------------------------------------------------------
    # function: list_row_w
    # write a column of data
    # input variables:
    # data : the list of the values of the data
    # loc_x (int): the row to write
    # sheet_name (string): the sheet to write the data
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def list_row_w(self,data,loc_x,sheet_name):
        assert type(data) is list
        work_sheet = self.find_sheet(sheet_name)
        for loc_y in range(0,len(data)):
            work_sheet.write(loc_x,loc_y+1,data[loc_y])
        pass

    # --------------------------------------------------------------------------------
    # function: matrix_w
    # write a matrix of data
    # input variables:
    # data : the matrix of the values of the data
    # sheet_name (string): the sheet to write the data
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def matrix_w(self,data,sheet_name):
        #work_sheet = self.find_sheet(sheet_name)
        for col in range(0,len(data)):
            self.list_column_w(data[col],col+1,sheet_name)

        pass

    # --------------------------------------------------------------------------------
    # function: save
    # save the data file
    # input variables:
    # bookname (string): the file name of the data
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def save(self,bookname=''):
        if bookname == '':
            self.wb.save(self.book_name+'.xls')
        else:
            self.wb.save(bookname+'.xls')