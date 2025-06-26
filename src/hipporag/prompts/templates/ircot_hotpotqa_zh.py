one_shot_ircot_demo_docs = (
    """维基百科标题: Milk and Honey (专辑)\nMilk and Honey 是约翰·列侬和小野洋子于1984年发行的专辑。继合集《The John Lennon Collection》之后，这是列侬的第八张也是最后一张录音室专辑，也是第一张列侬音乐的遗作发行，录制于他人生最后几个月中，在他们1980年专辑《Double Fantasy》的录制期间和之后。该专辑由小野洋子与Geffen唱片公司合作制作。\n\n"""
    """维基百科标题: 约翰·列侬博物馆\n约翰·列侬博物馆（ジョン・レノン・ミュージアム，Jon Renon Myūjiamu）是一座位于日本埼玉县埼玉市中央区埼玉超级竞技场内的博物馆。该博物馆成立的目的是保存约翰·列侬的生平和音乐生涯知识。它展示了列侬的遗孀小野洋子收藏的他的纪念品以及其他展品。博物馆于2000年10月9日，即列侬诞辰60周年纪念日开馆，并于2010年9月30日关闭，当时与小野洋子的展览合同到期。博物馆参观以小野洋子讲述的欢迎词和短片开始（日语配音，提供英语耳机），最后到达一个前卫风格的"冥想室"，里面摆满了面向移动文字和图像幻灯片的椅子。这个房间后面是一个礼品店，有约翰·列侬纪念品出售。\n\n"""
    """维基百科标题: Walls and Bridges\nWalls and Bridges 是英国音乐家约翰·列侬的第五张录音室专辑。该专辑于1974年9月26日在美国由苹果唱片公司发行，10月4日在英国发行。这张专辑在他与小野洋子分居18个月期间创作、录制和发行，记录了列侬"失落周末"时期的状态。《Walls and Bridges》是美国《Billboard》排行榜第一名专辑，并产生了两首热门单曲《Whatever Gets You thru the Night》和《#9 Dream》。第一首歌是列侬作为独唱艺术家在美国的第一首冠军歌曲，也是他一生中在美国或英国唯一登顶的单曲。\n\n"""
    """维基百科标题: Nobody Loves You (When You're Down and Out)\n《Nobody Loves You (When You're Down and Out)》是约翰·列侬创作的一首歌曲，收录在他1974年的专辑《Walls and Bridges》中。这首歌曲也收录在1986年的合集《Menlove Ave.》、1990年的套装《Lennon》、1998年的套装《John Lennon Anthology》、2005年的双碟合集以及2010年的套装《Gimme Some Truth》中。\n\n"""
    """维基百科标题: Give Peace a Chance\n《Give Peace a Chance》是约翰·列侬创作的一首反战歌曲（署名为列侬-麦卡特尼），与小野洋子在加拿大魁北克省蒙特利尔演出。1969年由塑料小野乐队在苹果唱片公司发行单曲（英国目录号Apple 13，美国目录号Apple 1809），这是列侬发行的第一首个人单曲，当时他仍是披头士乐队成员，在1970年代成为美国反战运动的颂歌。该曲在《Billboard》热门100排行榜上达到第14位，在英国单曲排行榜上达到第2位。\n"""
)


one_shot_ircot_demo = (
    f'{one_shot_ircot_demo_docs}'
    '\n\n问题: '
    f"Nobody Loves You 是由约翰·列侬创作并发行在哪张由苹果唱片公司发行的专辑上，该专辑是在他与小野洋子分居18个月期间创作、录制和发行的？"
    '\n思考: '
    f"由苹果唱片公司发行，在约翰·列侬与小野洋子分居18个月期间创作、录制和发行的专辑是 Walls and Bridges。Nobody Loves You 是约翰·列侬在 Walls and Bridges 专辑上创作的歌曲。所以答案是：Walls and Bridges。"
    '\n\n'
)

ircot_system = (
    '您是一个智能助手，擅长引导用户通过复杂的多跳推理来理解多个文档。这项任务通过演示来说明，每个演示包含一组文档、相关问题及其多跳推理思路。您的任务是为当前步骤生成一个思考过程，不要一次性生成所有思路！如果您认为已经到达最终步骤，请以"所以答案是："开头。'
    '\n\n'
    f'{one_shot_ircot_demo}'
)


prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"}
] 