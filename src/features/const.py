ID_COLS = ['TransactionID', 'TransactionDT']

V0_CAT_COLS = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',
               'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
               'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
               'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4', 'P_emaildomain',
               'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6',
               'M7', 'M8', 'M9']

V1_COLS = ['TransactionAmt', 'ProductCD',
           'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
           'addr1', 'addr2',
           'dist1', 'dist2',
           'P_emaildomain', 'R_emaildomain']
V1_CAT_COLS = ['ProductCD',
               'card4', 'card6',
               'P_emaildomain', 'R_emaildomain']
V1_CAT_COLS_FEATURES = ['ProductCD',
                        'card4', 'card6',
                        'P_emaildomain_first', 'P_emaildomain_rest', 'R_emaildomain_first', 'R_emaildomain_rest']
