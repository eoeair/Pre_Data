# KuaiRec: A Fully-observed Dataset for Recommender Systems (Density: Almost 100%)

## Data Descriptions

*KuaiRec* contains millions of user-item interactions as well as side information including the item categories and a social network. Six files are included in the download data: 

  ```shell
  KuaiRec
  ├── data
  │   ├── big_matrix.csv          
  │   ├── small_matrix.csv
  │   ├── social_network.csv
  │   ├── user_features.csv
  │   ├── item_daily_features.csv
  │   └── item_categories.csv
  │   └── kuairec_caption_category.csv
  ```

The statistics of the small matrix and big matrix in *KuaiRec*.

|                | #Users | #Items | #Interactions | Density |
| -------------- | :----: | :----: | :-----------: | :-----: |
| *small matrix* | 1,411  | 3,327  |   4,676,570   |  99.6%  |
| *big matrix*   | 7,176  | 10,728 |  12,530,806   |  16.3%  |

Note that the density of the small matrix is 99.6% instead of 100% because some users have explicitly indicated that they would not be willing to receive recommendations from certain authors. I.e., They blocked these videos.

#### 1. Descriptions of the fields in `big_matrix.csv` and `small_matrix.csv`. 

| Field Name:    | Description                                              | Type    | Example                   |
| -------------- | -------------------------------------------------------- | ------- | ------------------------- |
| user_id        | The ID of the user.                                      | int64   | 0                         |
| video_id       | The ID of the viewed video.                              | int64   | 3650                      |
| play_duration  | Time of video viewing of this interaction (millisecond). | int64   | 13838                     |
| video_duration | Time of this video (millisecond).                        | int64   | 10867                     |
| time           | Human-readable date for this interaction                 | str     | "2020-07-05 00:08:23.438" |
| date           | Date of this interaction                                 | int64   | 20200705                  |
| timestamp      | Unix timestamp                                           | float64 | 1593878903.438            |
| watch_ratio    | The video watching ratio (=play_duration/video_duration) | float64 | 1.273397                  |

The "watch_ratio" can be deemed as the label of the interaction. Note: there is no "like" signal for this dataset. If you need this binary signal in your scenarios, you can create it yourself. E.g., `like = 1 if watch_ratio > 2.0`.

#### 2. Descriptions of the fields in `social_network.csv`

| Field Name: | Description                                  | Type  | Example     |
| ----------- | -------------------------------------------- | ----- | ----------- |
| user_id     | The ID of the user.                          | int64 | 5352        |
| friend_list | The list of IDs of the friends of this user. | list  | [4202,7126] |

#### 3. Descriptions of the fields in `item_categories.csv`. 

| Field Name: | Description                     | Type  | Example |
| ----------- | ------------------------------- | ----- | ------- |
| video_id    | The ID of the video.            | int64 | 1       |
| feat        | The list of tags of this video. | list  | [27,9]  |

#### 4. Descriptions of the fields in `item_daily_features.csv`. (Added on 2022.05.16)

| Field Name:              | Description                                                                                                                    | Type    | Example       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------- |
| video_id                 | The ID of the video.                                                                                                           | int64   | 3784          |
| date                     | Date of the statistics of this video.                                                                                          | int64   | 20200730      |
| author_id                | The ID of the author of this video.                                                                                            | int64   | 441           |
| video_type               | Type of this video (NORMAL or AD).                                                                                             | str     | "NORMAL"      |
| upload_dt                | Upload date of this video.                                                                                                     | str     | "2020-07-08"  |
| upload_type              | The upload type of this video.                                                                                                 | str     | "ShortImport" |
| visible_status           | The visible state of this video on the APP now.                                                                                | str     | "public"      |
| video_duration           | The time duration of this duration (in milliseconds).                                                                          | float64 | 17200.0       |
| video_width              | The width of this video on the server.                                                                                         | int64   | 720           |
| video_height             | The height of this video on the server.                                                                                        | int64   | 1280          |
| music_id                 | Background music ID of this video.                                                                                             | int64   | 989206467     |
| video_tag_id             | The ID of the tag of this video.                                                                                               | int64   | 2522          |
| video_tag_name           | The name of the tag of this video.                                                                                             | string  | "祝福"        |
| show_cnt                 | The number of shows of this video **within this day (the same with all following fields)**                                     | int64   | 7716          |
| show_user_num            | The number of users who received the recommendation of this video.                                                             | int64   | 5256          |
| play_cnt                 | The number of plays.                                                                                                           | int64   | 7701          |
| play_user_num            | The number of users who play this video.                                                                                       | int64   | 5034          |
| play_duration            | The total time duration of playing this video (in milliseconds).                                                               | int64   | 138333346     |
| complete_play_cnt        | The number of complete plays. *complete play*: finishing playing the whole video, i.e., `#(play_duration >= video_duration)`.  | int64   | 3446          |
| complete_play_user_num   | The number of users who perform the *complete play*.                                                                           | int64   | 2033          |
| valid_play_cnt           | *valid play*: `play_duration >= video_duration if video_duration <= 7s`, or `play_duration > 7 if video_duration > 7s`.        | int64   | 5099          |
| valid_play_user_num      | The number of users who perform the *complete play*.                                                                           | int64   | 3195          |
| long_time_play_cnt       | *long time play*: `play_duration >= video_duration if video_duration <= 18s`, or `play_duration >=18 if video_duration > 18s`. | int64   | 3299          |
| long_time_play_user_num  | The number of users who perform the *long time play*.                                                                          | int64   | 1940          |
| short_time_play_cnt      | *short time play*: `play_duration < min(3s, video_duration)`.                                                                  | int64   | 1538          |
| short_time_play_user_num | The number of users who perform the *short time play*.                                                                         | int64   | 1190          |
| play_progress            | The average video playing ratio (`=play_duration/video_duration`)                                                              | int64   | 0.579695      |
| comment_stay_duration    | Total time of staying in the comments section                                                                                  | int64   | 467865        |
| like_cnt                 | Total likes                                                                                                                    | int64   | 659           |
| like_user_num            | The number of users who hit the "like" button.                                                                                 | int64   | 657           |
| click_like_cnt           | The number of the "like" resulted from double click                                                                            | int64   | 496           |
| double_click_cnt         | The number of users who double-click the video.                                                                                | int64   | 163           |
| cancel_like_cnt          | The number of likes that are canceled by users.                                                                                | int64   | 15            |
| cancel_like_user_num     | The number of users who cancel their likes.                                                                                    | int64   | 15            |
| comment_cnt              | The number of comments within this day.                                                                                        | int64   | 13            |
| comment_user_num         | The number of users who comment on this video.                                                                                 | int64   | 12            |
| direct_comment_cnt       | The number of direct comments (depth=1).                                                                                       | int64   | 13            |
| reply_comment_cnt        | The number of reply comments (depth>1).                                                                                        | int64   | 0             |
| delete_comment_cnt       | The number of deleted comments.                                                                                                | int64   | 0             |
| delete_comment_user_num  | The number of users who delete their comments.                                                                                 | int64   | 0             |
| comment_like_cnt         | The number of comment likes.                                                                                                   | int64   | 2             |
| comment_like_user_num    | The number of users who like the comments.                                                                                     | int64   | 2             |
| follow_cnt               | The number of increased follows from this video.                                                                               | int64   | 151           |
| follow_user_num          | The number of users who follow the author of this video due to this video.                                                     | int64   | 151           |
| cancel_follow_cnt        | The number of decreased follows from this video.                                                                               | int64   | 0             |
| cancel_follow_user_num   | The number of users who cancel their following of the author of this video due to this video.                                  | int64   | 0             |
| share_cnt                | The times of sharing this video.                                                                                               | int64   | 1             |
| share_user_num           | The number of users who share this video.                                                                                      | int64   | 1             |
| download_cnt             | The times of downloading this video.                                                                                           | int64   | 2             |
| download_user_num        | The number of users who download this video.                                                                                   | int64   | 2             |
| report_cnt               | The times of reporting this video.                                                                                             | int64   | 0             |
| report_user_num          | The number of users who report this video.                                                                                     | int64   | 0             |
| reduce_similar_cnt       | The times of reducing similar content of this video.                                                                           | int64   | 2             |
| reduce_similar_user_num  | The number of users who choose to reduce similar content of this video.                                                        | int64   | 2             |
| collect_cnt              | The times of adding this video to favorite videos.                                                                             | int64   | 0             |
| collect_user_num         | The number of users who add this video to their favorite videos.                                                               | int64   | 0             |
| cancel_collect_cnt       | The times of removing this video from favorite videos.                                                                         | int64   | 0             |
| cancel_collect_user_num  | The number of users who remove this video from their favorite videos                                                           | int64   | 0             |


#### 5. Descriptions of the fields in `user_features.csv` (Added on 2022.05.16)

| Field Name:           | Description                                                                                                                                                  | Type  | Example       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ------------- |
| user_id               | The ID of the user.                                                                                                                                          | int64 | 0             |
| user_active_degree    | In the set of {'high_active', 'full_active', 'middle_active', 'UNKNOWN'}.                                                                                    | str   | "high_active" |
| is_lowactive_period   | Is this user in its low active period                                                                                                                        | int64 | 0             |
| is_live_streamer      | Is this user a live streamer？                                                                                                                               | int64 | 0             |
| is_video_author       | Has this user uploaded any video？                                                                                                                           | int64 | 0             |
| follow_user_num       | The number of users that this user follows.                                                                                                                  | int64 | 5             |
| follow_user_num_range | The range of the number of users that this user follows. In the set of {'0', '(0,10]', '(10,50]', '(100,150]', '(150,250]', '(250,500]', '(50,100]', '500+'} | str   | "(0,10]"      |
| fans_user_num         | The number of the fans of this user.                                                                                                                         | int64 | 0             |
| fans_user_num_range   | The range of the number of fans of this user. In the set of {'0', '[1,10)', '[10,100)', '[100,1k)', '[1k,5k)',  '[5k,1w)', '[1w,10w)'}                       | str   | "0"           |
| friend_user_num       | The number of friends that this user has.                                                                                                                    | int64 | 0             |
| friend_user_num_range | The range of the number of friends that this user has.  In the set of {'0', '[1,5)', '[5,30)', '[30,60)', '[60,120)', '[120,250)', '250+'}                   | str   | "0"           |
| register_days         | The days since this user has registered.                                                                                                                     | int64 | 107           |
| register_days_range   | The range of the registered days. In the set of {'15-30', '31-60', '61-90', '91-180', '181-365', '366-730', '730+'}.                                         | str   | "61-90"       |
| onehot_feat0          | An encrypted feature of the user. Each value indicate the position of "1" in the one-hot vector. Range: {0,1}                                                | int64 | 0             |
| onehot_feat1          | An encrypted feature. Range: {0, 1, ..., 7}                                                                                                                  | int64 | 1             |
| onehot_feat2          | An encrypted feature. Range: {0, 1, ..., 29}                                                                                                                 | int64 | 17            |
| onehot_feat3          | An encrypted feature. Range: {0, 1, ..., 1075}                                                                                                               | int64 | 638           |
| onehot_feat4          | An encrypted feature. Range: {0, 1, ..., 11}                                                                                                                 | int64 | 2             |
| onehot_feat5          | An encrypted feature. Range: {0, 1, ..., 9}                                                                                                                  | int64 | 0             |
| onehot_feat6          | An encrypted feature. Range: {0, 1, 2}                                                                                                                       | int64 | 1             |
| onehot_feat7          | An encrypted feature. Range: {0, 1, ..., 46}                                                                                                                 | int64 | 6             |
| onehot_feat8          | An encrypted feature. Range: {0, 1, ..., 339}                                                                                                                | int64 | 184           |
| onehot_feat9          | An encrypted feature. Range: {0, 1, ..., 6}                                                                                                                  | int64 | 6             |
| onehot_feat10         | An encrypted feature. Range: {0, 1, ..., 4}                                                                                                                  | int64 | 3             |
| onehot_feat11         | An encrypted feature. Range: {0, 1, ..., 2}                                                                                                                  | int64 | 0             |
| onehot_feat12         | An encrypted feature. Range: {0, 1}                                                                                                                          | int64 | 0             |
| onehot_feat13         | An encrypted feature. Range: {0, 1}                                                                                                                          | int64 | 0             |
| onehot_feat14         | An encrypted feature. Range: {0, 1}                                                                                                                          | int64 | 0             |
| onehot_feat15         | An encrypted feature. Range: {0, 1}                                                                                                                          | int64 | 0             |
| onehot_feat16         | An encrypted feature. Range: {0, 1}                                                                                                                          | int64 | 0             |
| onehot_feat17         | An encrypted feature. Range: {0, 1}                                                                                                                          | int64 | 0             |


#### 6. Descriptions of the caption and category fields in `kuairec_caption_category.csv` (Added on 2024.06.02)


| Field Name:                | Description                                            | Type  | Example                                                      |
| -------------------------- | ------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| video_id                   | The ID of the video                                    | int64 | 2418                                                         |
| manual_cover_text          | 封面文字 (added by its author)                         | str   | "被小可爱发现了"                                             |
| caption                    | 简介标题 (added by its author)                         | str   | "这是什么狗狗，这么可爱真的可以这么遛吗？#喜欢的双击加关注 #直播 #博美俊介 #萌宠驾到" |
| topic_tag                  | Tags of the topics of this video (added by its author) | str   | "[博美俊介,喜欢的双击加关注,直播,萌宠驾到]"                  |
| first_level_category_id    | First-level category ID                                | int64 | 17                                                           |
| first_level_category_name  | First-level category name                              | str   | "宠物"                                                       |
| second_level_category_id   | Second-level category ID                               | int64 | 233                                                          |
| second_level_category_name | Second-level category name                             | str   | "宠物日常记录"                                               |
| third_level_category_id    | Thrid-level category ID                                | int64 | 1169                                                         |
| third_level_category_name  | Third-level category name                              | str   | "宠物狗"                                                     |