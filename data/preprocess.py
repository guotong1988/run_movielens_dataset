debug = False

f = open("ratings.csv", encoding="utf-8", mode="r")

ratings = []

for i, line in enumerate(f):
    if i == 0:
        continue
    splits = line.strip().split(",")
    userId = splits[0]
    movieId = splits[1]
    rating = splits[2]
    ratings.append(splits)
    if debug and i > 1000:
        break

print("ratings", len(ratings))

tag_to_id = {}
f = open("genome-tags.csv", encoding="utf-8", mode="r")
for i, line in enumerate(f):
    if i == 0:
        continue
    splits = line.strip().split(",")
    tagId = splits[0]
    tag = splits[1].lower()
    tag = tag.replace(" ", "-")
    tag_to_id[tag] = tagId

print("tag_to_id", len(tag_to_id))

f = open("movies.csv", encoding="utf-8", mode="r")

moive_to_top5tagids = {}
for i, line in enumerate(f):
    if i == 0:
        continue
    splits = line.strip().split(",")
    movieId = splits[0]
    title = splits[1]
    splits2 = splits[-1].split("|")
    result_tagids = []
    for tag in splits2:
        tag = tag.lower()
        if tag != 'imax' and tag != "(no genres listed)":
            result_tagids.append(tag_to_id[tag])
        if tag == "(no genres listed)":
            result_tagids.append("0")

    moive_to_top5tagids[movieId] = result_tagids
    if debug and i > 1000:
        break

print("moive_to_top5tagids", len(moive_to_top5tagids))

f = open("genome-scores.csv", encoding="utf-8", mode="r")

movie_to_top10tagids = {}
movie_to_top10scores = {}
for i, line in enumerate(f):
    if i == 0:
        continue
    splits = line.strip().split(",")
    movieId = splits[0]
    tagId = int(splits[1])
    tagScore = float(splits[2])

    if movieId in movie_to_top10tagids:
        top10tagId = movie_to_top10tagids[movieId]
        top10score = movie_to_top10scores[movieId]
        if tagScore > min(top10score):
            min_index = top10score.index(min(top10score))
            top10tagId[min_index] = tagId
            top10score[min_index] = tagScore
        movie_to_top10tagids[movieId] = top10tagId
        movie_to_top10scores[movieId] = top10score
    else:
        top10score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        top10tagId = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if tagScore > min(top10score):
            min_index = top10score.index(min(top10score))
            top10tagId[min_index] = tagId
            top10score[min_index] = tagScore
        movie_to_top10tagids[movieId] = top10tagId
        movie_to_top10scores[movieId] = top10score
    if i > 10000 and debug:
        break

print("moive_to_top10tagids", len(movie_to_top10tagids))
user_to_tag_to_score = {}

f = open("ratings.csv", encoding="utf-8", mode="r")
for i, line in enumerate(f):
    if i == 0:
        continue
    splits = line.strip().split(",")
    userId = splits[0]
    movieId = splits[1]
    movieRating = splits[2]

    tagScores1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagIds1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if movieId in movie_to_top10tagids:
        tagScores1 = movie_to_top10scores[movieId]
        tagIds1 = movie_to_top10tagids[movieId]
    tagIds2 = [0, 0, 0, 0, 0]
    if movieId in moive_to_top5tagids:
        tagIds2 = moive_to_top5tagids[movieId]
    if userId in user_to_tag_to_score:
        tag_to_score = user_to_tag_to_score[userId]
    else:
        tag_to_score = {}
    for index, tag in enumerate(tagIds1):
        if tag != 0:
            if tag in tag_to_score:
                score_now = tag_to_score[tag]
                score_now += tagScores1[index] * 5
                tag_to_score[tag] = score_now
            else:
                score_now = 0.0
                score_now += tagScores1[index] * 5
                tag_to_score[tag] = score_now

    for index, tag in enumerate(tagIds2):
        if tag != 0:
            if tag in tag_to_score:
                score_now = tag_to_score[tag]
                score_now += float(movieRating)
                tag_to_score[tag] = score_now
            else:
                score_now = 0.0
                score_now += float(movieRating)
                tag_to_score[tag] = score_now

    user_to_tag_to_score[userId] = tag_to_score
    if debug and i > 10000:
        break

print("user_to_tag_to_score", len(user_to_tag_to_score))

f = open("train.tsv", mode="w", encoding="utf-8")

sparse_feature_names = ['C' + str(i) for i in range(1, 31)]
f.write("label," + ",".join(sparse_feature_names) + "\n")

for splits in ratings:
    userId = splits[0]
    movieId = splits[1]
    rating = splits[2]

    tag_to_score = user_to_tag_to_score[userId]
    top10tagId_of_user = []

    tag_to_score2 = {k: v for k, v in sorted(tag_to_score.items(), key=lambda item: -item[1])}

    for key in tag_to_score2:
        top10tagId_of_user.append(str(key))

    while len(top10tagId_of_user) < 15:
        top10tagId_of_user.append("0")

    movie_tags = set()
    tagIds1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if movieId in movie_to_top10tagids:
        tagIds1 = movie_to_top10tagids[movieId]
    tagIds2 = [0, 0, 0, 0, 0]
    if movieId in moive_to_top5tagids:
        tagIds2 = moive_to_top5tagids[movieId]

    for index, tag in enumerate(tagIds1):
        movie_tags.add(str(tag))

    for index, tag in enumerate(tagIds2):
        movie_tags.add(str(tag))

    movie_tags = list(movie_tags)
    while len(movie_tags) < 15:
        movie_tags.append("0")

    f.write(str(float(rating) / 5) + "," + ",".join(top10tagId_of_user[:15]) + "," + ",".join(movie_tags[:15]) + "\n")
