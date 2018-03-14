# class Solution(object):
#     def threeSumClosest(self, nums, target):
#
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: int
#         """
#
#         indexToRemove = 0
#         indexToAdd = 0
#
#         if len(nums):
#             closestSum = nums[0] + nums[1] + nums[2]
#             indices = [0, 1, 2]
#             for i in range(3, len(nums)):
#                 tempSum = closestSum
#                 intermediateSum = tempSum
#
#                 for index in indices:
#                     if abs(target - (tempSum + nums[i] - nums[index])) < abs(target - intermediateSum):
#                         intermediateSum = tempSum + nums[i] - nums[index]
#                         indexToRemove = index
#                         indexToAdd = i
#
#
#
#                 tempSum = intermediateSum
#
#                 if indexToAdd or indexToRemove:
#                     closestSum = intermediateSum
#                     indices.remove(indexToRemove)
#                     indices.append(indexToAdd)
#                     indexToRemove = 0
#                     indexToAdd = 0
#
#
#             return closestSum
#
#
# def stringToIntegerList(input):
#     return json.loads(input)
#
#
# def stringToInt(input):
#     return int(input)
#
#
# def intToString(input):
#     if input is None:
#         input = 0
#     return str(input)
#
#
# def main():
#     import sys
#     def readlines():
#         for line in sys.stdin:
#             yield line.strip('\n')
#
#     lines = readlines()
#     # while True:
#     #     try:
#     #         line = lines.next()
#     #         nums = stringToIntegerList(line)
#     #         line = lines.next()
#     #         target = stringToInt(line)
#     #
#     #         ret = Solution().threeSumClosest(nums, target)
#     #
#     #         out = intToString(ret)
#     #         print out
#     #     except StopIteration:
#     #         break
#
#     ret = Solution().threeSumClosest([1, 2, 4, 8, 16, 32, 64, 128], 82)
#
#     out = intToString(ret)
#     print out
#
# if __name__ == '__main__':
#     main()


class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """

        anagrams = dict()
        if len(strs):

            for string in strs:

                sortedkey = sorted(string)
                if not anagrams.has_key(str(sortedkey)):
                    anagrams.__setitem__(str(sorted(string)), [])
                anagrams.get(str(sorted(string))).append(string)

        print(anagrams)

        anagramsList = []
        for key in anagrams.keys():
            anagramsList.append(anagrams.get(key))

        return anagramsList

def stringToStringArray(input):
    return json.loads(input)


def string2dArrayToString(input):
    return json.dumps(input)


def main():
    import sys
    def readlines():
        for line in sys.stdin:
            yield line.strip('\n')
    #
    # lines = readlines()
    # while True:
    #     try:
    #         line = lines.next()
    #         strs = stringToStringArray(line)

    ret = Solution().groupAnagrams(["eat","tea","tan","ate","nat","bat"])

    out = string2dArrayToString(ret)
    print out
        # except StopIteration:
        #     break


if __name__ == '__main__':
    main()