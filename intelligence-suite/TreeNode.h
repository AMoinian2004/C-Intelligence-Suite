#ifndef TREENODE_H
#define TREENODE_H

struct TreeNode {
    int featureIndex;
    double splitValue;
    double predictedClass;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int feature = -1, double split = 0, double predClass = 0, TreeNode* l = nullptr, TreeNode* r = nullptr) 
        : featureIndex(feature), splitValue(split), predictedClass(predClass), left(l), right(r) {}
};

#endif // TREENODE_H
