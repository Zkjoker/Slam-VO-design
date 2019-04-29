#include "myslam/map.h"

namespace myslam
{

void Map::insertKeyFrame ( Frame::Ptr frame )
{
    cout<<"Key frame size = "<<keyframes_.size()<<endl;
    //关键帧中没有这一帧，就插入（id号应该是按大小排序的，id号比当前关键帧中的帧号都大，find到的就应该是结尾
    if ( keyframes_.find(frame->id_) == keyframes_.end() )
    {
        keyframes_.insert( make_pair(frame->id_, frame) );  //没这个索引，插入帧，并且把id和帧配对起来，
    }
    else
    {
        keyframes_[ frame->id_ ] = frame;  //find到的不是结尾，说明map数据结构中有这个索引，中间补插入一张，直接插入
    }
}

void Map::insertMapPoint ( MapPoint::Ptr map_point )  //和上面的同理
{
    if ( map_points_.find(map_point->id_) == map_points_.end() )
    {
        map_points_.insert( make_pair(map_point->id_, map_point) );
    }
    else 
    {
        map_points_[map_point->id_] = map_point;
    }
}


}