#include <iostream>
#include <string>
#include <pcap.h>

using namespace std;

int main(int argc,char*argv[])
{
    //get file name
    string file = "";
    //error buffer
    char errbuff[PCAP_ERRBUF_SIZE]

    //open file and store result in pcap_t
    pcap_t* pcap= pcap_open_offline(file.c_str(),errbuff);

    //header object
    cont u_char *data;

    //print
    u_int packetCount=0;
    while(int returnValue= pcap_next_ex(pcap,&header,&data)>=0)
    {
        stdout << "Packet # " << ++packetCount << endl;
        
    }
}
