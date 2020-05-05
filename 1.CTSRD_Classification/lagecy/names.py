dic = ['限速5km/h', '限速15km/h', '限速30km/h', '限速40km/h', '限速50km/h',
       '限速60km/h', '限速70km/h', '限速80km/h', '禁止直行和左转', '禁止直行和右转',
       '禁止直行', '禁止左转', '禁止左转和右转', '禁止右转', '禁止超车',
       '禁止调头', '禁止机动车', '禁止鸣笛', '解除40km/h限速', '解除50km/h限速',
       '直行和右转', '直行', '左转', '左转和右转', '右转',
       '左侧行驶', '右侧行驶', '环岛', '机动车道', '鸣笛',
       '非机动车道', '调头', '注意左右绕行', '注意信号灯', '注意危险',
       '注意人行横道', '注意非机动车', '注意学校', '注意向右急转弯', '注意向左急转弯',
       '注意下坡', '注意上坡', '注意慢行', '注意向右T型交叉', '注意向左T型交叉',
       '注意村镇', '注意反向弯路', '注意无人看守铁道路口', '注意施工', '注意连续弯道',
       '注意有人看守铁道路口', '注意事故易发路段', '停车让行', '禁止通行', '禁止车辆临时或长时停放',
       '禁止驶入', '减速让行', '停车检查']


names = {'停车检查': 0, '停车让行': 1, '减速让行': 2, '右侧行驶': 3, '右转': 4,
         '左侧行驶': 5, '左转': 6, '左转和右转': 7, '机动车道': 8, '注意上坡': 9,
         '注意下坡': 10, '注意事故易发路段': 11, '注意人行横道': 12, '注意信号灯': 13, '注意危险': 14,
         '注意反向弯路': 15, '注意向右T型交叉': 16, '注意向右急转弯': 17, '注意向左T型交叉': 18, '注意向左急转弯': 19,
         '注意学校': 20, '注意左右绕行': 21, '注意慢行': 22, '注意施工': 23, '注意无人看守铁道路口': 24,
         '注意有人看守铁道路口': 25, '注意村镇': 26, '注意连续弯道': 27, '注意非机动车': 28, '环岛': 29,
         '直行': 30, '直行和右转': 31, '禁止右转': 32, '禁止左转': 33, '禁止左转和右转': 34,
         '禁止机动车': 35, '禁止直行': 36, '禁止直行和右转': 37, '禁止直行和左转': 38, '禁止调头': 39,
         '禁止超车': 40, '禁止车辆临时或长时停放': 41, '禁止通行': 42, '禁止驶入': 43, '禁止鸣笛': 44,
         '解除40km/h限速': 45, '解除50km/h限速': 46, '调头': 47, '限速15km/h': 48, '限速30km/h': 49,
         '限速40km/h': 50, '限速50km/h': 51, '限速5km/h': 52, '限速60km/h': 53, '限速70km/h': 54,
         '限速80km/h': 55, '非机动车道': 56, '鸣笛': 57}